!  THIS VERSION: GALAHAD 5.1 - 2024-11-23 AT 15:35 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_TRB_test
   USE GALAHAD_USERDATA_precision
   USE GALAHAD_TRB_precision                       ! double precision version
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( TRB_control_type ) :: control
   TYPE ( TRB_inform_type ) :: inform
   TYPE ( TRB_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
!  EXTERNAL :: FUN, GRAD, HESS, HESSPROD, PREC
   INTEGER ( KIND = ip_ ) :: i, s, scratch_out
   logical :: alive
   REAL ( KIND = rp_ ), PARAMETER :: p = 4.0_rp_
   REAL ( KIND = rp_ ) :: dum
! start problem data
   nlp%n = 1 ; nlp%H%ne = 1                     ! dimensions
   ALLOCATE( nlp%X( nlp%n ), nlp%X_l( nlp%n ), nlp%X_u( nlp%n ),               &
             nlp%G( nlp%n ) )
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%H%val( nlp%H%ne ), nlp%H%row( nlp%H%ne ),                     &
             nlp%H%col( nlp%H%ne ) )
   nlp%H%row = (/ 1 /) ; nlp%H%col = (/ 1 /)

! problem data complete

   IF ( .FALSE. ) GO TO 10

!  ================
!  error exit tests
!  ================

   WRITE( 6, "( /, ' error exit tests ' )" )

!  tests for s = - 1 ... - 40

   DO s = 1, 40
     IF ( s > 24 .AND. s <= 40 ) CYCLE
     IF ( s == - GALAHAD_error_allocate ) CYCLE
     IF ( s == - GALAHAD_error_deallocate ) CYCLE
!    IF ( s == - GALAHAD_error_restrictions ) CYCLE
!    IF ( s == - GALAHAD_error_bad_bounds ) CYCLE
     IF ( s == - GALAHAD_error_primal_infeasible ) CYCLE
     IF ( s == - GALAHAD_error_dual_infeasible ) CYCLE
!    IF ( s == - GALAHAD_error_unbounded ) CYCLE
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
     CALL TRB_initialize( data, control, inform )! Initialize control parameters
     CALL WHICH_sls( control )
     control%out = 0 ; control%error = 0
!    control%print_level = 1
     inform%status = 1                           ! set for initial entry
     nlp%n = 1
     nlp%X = 1.0_rp_                              ! start from one
     nlp%X_l = 0.0_rp_ ; nlp%X_u = 2.0_rp_         ! search in [0,2]
     control%hessian_available = .FALSE.         ! Hessian prods will be used

     IF ( s == - GALAHAD_error_restrictions ) THEN
       nlp%n = 0
     ELSE IF ( s == - GALAHAD_error_bad_bounds ) THEN
       nlp%X_u = - 2.0_rp_
     ELSE IF ( s == - GALAHAD_error_unbounded ) THEN
       nlp%X_l = - ( 10.0_rp_ ) ** 30 ; nlp%X_u = ( 10.0_rp_ ) ** 30
       control%maximum_radius = ( 10.0_rp_ ) ** 20
     ELSE IF ( s == - GALAHAD_error_max_iterations ) THEN
       control%maxit = 0
     ELSE IF ( s == - GALAHAD_error_preconditioner ) THEN
       control%norm = - 3                        ! User's preconditioner
     ELSE IF ( s == - GALAHAD_error_cpu_limit ) THEN
       control%cpu_time_limit = 0.0_rp_
     END IF
     DO                                           ! Loop to solve problem
       CALL TRB_solve( nlp, control, inform, data, userdata )
!write(6,*) ' status ', inform%status
       SELECT CASE ( inform%status )              ! reverse communication
       CASE ( 2 )                                 ! Obtain the objective
         nlp%f = - nlp%X( 1 ) ** 2
         data%eval_status = 0                     ! record successful evaluation
         IF ( control%alive_unit > 0 .AND. s == 40 ) THEN
           INQUIRE( FILE = control%alive_file, EXIST = alive )
           IF ( alive .AND. control%alive_unit > 0 ) THEN
             OPEN( control%alive_unit, FILE = control%alive_file,              &
                   FORM = 'FORMATTED', STATUS = 'UNKNOWN' )
             REWIND control%alive_unit
             CLOSE( control%alive_unit, STATUS = 'DELETE' )
           END IF
         END IF
         IF ( s == - GALAHAD_error_cpu_limit ) THEN
           dum = 0.0_rp_
           DO i = 1, 10000000
             dum = dum + 0.0000001_rp_ * i / ( i + 1 )
           END DO
           nlp%f = ( nlp%f + dum ) - dum
         END IF
       CASE ( 3 )                                 ! Obtain the gradient
         nlp%G( 1 ) = - 2.0_rp_ * nlp%X( 1 )
         data%eval_status = 0                     ! record successful evaluation
       CASE ( 5 )                                 ! Obtain Hessian-vector prod
         data%U( 1 ) = data%U( 1 ) - 2.0_rp_ * data%V( 1 )
         data%eval_status = 0                     ! record successful evaluation
       CASE ( 6 )                                 ! Apply the preconditioner
         data%U( 1 ) = - data%V( 1 )
!        data%eval_status = 0                     ! record successful evaluation
       CASE ( 7 )                                 ! Obtain sparse Hess-vec prod
         data%HP( 1 ) = - 2.0_rp_ * data%P( 1 )
         data%nnz_hp = 1
         data%INDEX_nz_hp( 1 ) = 1
         data%eval_status = 0                     ! record successful evaluation
       CASE DEFAULT                               ! Terminal exit from loop
         EXIT
       END SELECT
     END DO
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &     F6.1, ' status = ', I6 )" ) i, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( I2, ': TRB_solve exit status = ', I6 ) " ) s, inform%status
     END IF

     CALL TRB_terminate( data, control, inform )  ! delete internal workspace
   END DO

   control%subproblem_direct = .TRUE.         ! Use a direct method
   CALL TRB_solve( nlp, control, inform, data, userdata,                       &
                   eval_F = FUN, eval_G = GRAD, eval_H = HESS )

10 continue
   DEALLOCATE( nlp%X, nlp%X_l, nlp%x_u, nlp%G )
   DEALLOCATE( nlp%H%val, nlp%H%row, nlp%H%col )

!  =========================
!  test of available options
!  =========================

! start problem data
   nlp%n = 3 ; nlp%H%ne = 5                  ! dimensions
   ALLOCATE( nlp%X( nlp%n ), nlp%X_l( nlp%n ), nlp%X_u( nlp%n ),               &
             nlp%G( nlp%n ) )
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%H%val( nlp%H%ne ), nlp%H%row( nlp%H%ne ),                     &
             nlp%H%col( nlp%H%ne ) )
   nlp%H%row = (/ 1, 3, 2, 3, 3 /)           ! Hessian H
   nlp%H%col = (/ 1, 1, 2, 2, 3 /)           ! NB lower triangle
! problem data complete
   ALLOCATE( userdata%real( 1 ) )             ! Allocate space to hold parameter
   userdata%real( 1 ) = p                     ! Record parameter, p

   WRITE( 6, "( /, ' test of availible options ', / )" )

!  DO i = 1, 1
   DO i = 1, 7
     CALL TRB_initialize( data, control, inform )! Initialize control parameters
     CALL WHICH_sls( control )
!    control%print_level = 1
     inform%status = 1                        ! set for initial entry
     nlp%X = 1.0_rp_                           ! start from one
     nlp%X_l = -2.5_rp_ ; nlp%X_u = 2.0_rp_      ! search in [0,2]

     IF ( i == 1 ) THEN
       ALLOCATE( nlp%VNAMES( nlp%n ) )
       nlp%VNAMES( 1 ) = 'X1' ; nlp%VNAMES( 2 ) = 'X2' ; nlp%VNAMES( 3 ) = 'X3'
       OPEN( NEWUNIT = scratch_out, STATUS = 'SCRATCH' )
       control%out = scratch_out ; control%error = scratch_out
!      control%print_level = 1
       control%print_level = 101
       control%print_gap = 2 ; control%stop_print = 5
       control%psls_control%out = scratch_out
       control%psls_control%error = scratch_out
       control%psls_control%print_level = 1
       control%trs_control%out = scratch_out
       control%trs_control%error = scratch_out
       control%trs_control%print_level = 1
       control%subproblem_direct = .TRUE.         ! Use a direct method
       CALL TRB_solve( nlp, control, inform, data, userdata,                   &
                       eval_F = FUN, eval_G = GRAD, eval_H = HESS )
       CLOSE( UNIT = scratch_out )
       DEALLOCATE( nlp%VNAMES )
     ELSE IF ( i == 2 ) THEN
       OPEN( NEWUNIT = scratch_out, STATUS = 'SCRATCH' )
       control%out = scratch_out ; control%error = scratch_out
       control%print_level = 101
       control%print_gap = 2 ; control%stop_print = 5
       control%norm = 2
       control%psls_control%out = scratch_out
       control%psls_control%error = scratch_out
       control%psls_control%print_level = 1
       control%trs_control%out = scratch_out
       control%trs_control%error = scratch_out
       control%trs_control%print_level = 1
       CALL TRB_solve( nlp, control, inform, data, userdata,                   &
                       eval_F = FUN, eval_G = GRAD,  eval_H = HESS )
       CLOSE( UNIT = scratch_out )
     ELSE IF ( i == 3 ) THEN
       control%norm = 3
       control%error = - 1
       CALL TRB_solve( nlp, control, inform, data, userdata,                   &
                       eval_F = FUN, eval_G = GRAD, eval_H = HESS )
     ELSE IF ( i == 4 ) THEN
       control%norm = 5
       control%error = - 1
       CALL TRB_solve( nlp, control, inform, data, userdata,                   &
                       eval_F = FUN, eval_G = GRAD,  eval_H = HESS )
     ELSE IF ( i == 5 ) THEN
       control%model = - 2
       CALL TRB_solve( nlp, control, inform, data, userdata,                   &
                       eval_F = FUN, eval_G = GRAD,  eval_H = HESS )
     ELSE IF ( i == 6 ) THEN
       control%model = 1
       control%maxit = 1000
       CALL TRB_solve( nlp, control, inform, data, userdata,                   &
                       eval_F = FUN, eval_G = GRAD )
     ELSE IF ( i == 7 ) THEN
       control%model = 3
       control%maxit = 1000
       CALL TRB_solve( nlp, control, inform, data, userdata,                   &
                       eval_F = FUN, eval_G = GRAD )
     END IF
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F6.1, ' status = ', I6 )" ) i, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( I2, ': TRB_solve exit status = ', I6 ) " ) i, inform%status
     END IF

     CALL TRB_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( nlp%X, nlp%X_l, nlp%x_u, nlp%G )
   DEALLOCATE( nlp%H%val, nlp%H%row, nlp%H%col, userdata%real )

!  ============================
!  full test of generic problem
!  ============================

! start problem data
   nlp%n = 3 ; nlp%H%ne = 5                  ! dimensions
   ALLOCATE( nlp%X( nlp%n ), nlp%X_l( nlp%n ), nlp%X_u( nlp%n ),               &
             nlp%G( nlp%n ) )
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%H%val( nlp%H%ne ), nlp%H%row( nlp%H%ne ),                     &
                        nlp%H%col( nlp%H%ne ) )
   nlp%H%row = (/ 1, 3, 2, 3, 3 /)            ! Hessian H
   nlp%H%col = (/ 1, 1, 2, 2, 3 /)            ! NB lower triangle
! problem data complete
   ALLOCATE( userdata%real( 1 ) )             ! Allocate space to hold parameter
   userdata%real( 1 ) = p                     ! Record parameter, p

   WRITE( 6, "( /, ' full test of generic problems ', / )" )

   DO i = 1, 6
     CALL TRB_initialize( data, control, inform )! Initialize control parameters
     CALL WHICH_sls( control )
!    control%print_level = 1
     inform%status = 1                          ! set for initial entry
     nlp%X = 1.0_rp_                             ! start from one
     nlp%X_l = 0.0_rp_ ; nlp%X_u = 2.0_rp_        ! search in [0,2]

     IF ( i == 1 ) THEN
       control%subproblem_direct = .TRUE.       ! Use a direct method
       CALL TRB_solve( nlp, control, inform, data, userdata,                   &
                       eval_F = FUN, eval_G = GRAD, eval_H = HESS )
     ELSE IF ( i == 2 ) THEN
       control%hessian_available = .FALSE.      ! Hessian products will be used
       CALL TRB_solve( nlp, control, inform, data, userdata,                   &
                       eval_F = FUN, eval_G = GRAD, eval_HPROD = HESSPROD,     &
                       eval_SHPROD = SHESSPROD )
     ELSE IF ( i == 3 ) THEN
       control%hessian_available = .FALSE.      ! Hessian products will be used
       control%norm = - 3                       ! User's preconditioner
       CALL TRB_solve( nlp, control, inform, data, userdata, eval_F = FUN,     &
              eval_G = GRAD, eval_HPROD = HESSPROD, eval_PREC = PREC,          &
              eval_SHPROD = SHESSPROD )
     ELSE IF ( i == 4 .OR. i == 5 .OR. i == 6 ) THEN
       IF ( i == 4 ) THEN
         control%subproblem_direct = .TRUE.         ! Use a direct method
       ELSE
         control%hessian_available = .FALSE.        ! Hessian prods will be used
       END IF
       IF ( i == 6 ) control%norm = - 3             ! User's preconditioner
       DO                                           ! Loop to solve problem
         CALL TRB_solve( nlp, control, inform, data, userdata )
         SELECT CASE ( inform%status )            ! reverse communication
         CASE ( 2 )                               ! Obtain the objective
           CALL FUN( data%eval_status, nlp%X( : nlp%n ), userdata, nlp%f )
         CASE ( 3 )                               ! Obtain the gradient
           CALL GRAD( data%eval_status, nlp%X( : nlp%n ), userdata,            &
                      nlp%G( : nlp%n ) )
         CASE ( 4 )                               ! Obtain the Hessian
           CALL HESS( data%eval_status, nlp%X( : nlp%n ), userdata,            &
                      nlp%H%val( : nlp%H%ne ) )
         CASE ( 5 )                              ! Obtain Hessian-vector prod
           CALL HESSPROD( data%eval_status, nlp%X( : nlp% n ), userdata,       &
                          data%U, data%V )
         CASE ( 6 )                              ! Apply the preconditioner
           CALL PREC( data%eval_status, nlp%X( : nlp%n ), userdata,            &
                          data%U, data%V )
         CASE ( 7 )                              ! Obtain sparse Hess-vec prod
           CALL SHESSPROD( data%eval_status, nlp%X( : nlp%n ), userdata,       &
                           data%nnz_p_u - data%nnz_p_l + 1,                    &
                           data%INDEX_nz_p(data%nnz_p_l:data%nnz_p_u), data%P, &
                           data%nnz_hp, data%INDEX_nz_hp, data%HP )
         CASE DEFAULT                            ! Terminal exit from loop
           EXIT
         END SELECT
       END DO
     ELSE
     END IF
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &     F6.1, ' status = ', I6 )" ) i, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( I2, ': TRB_solve exit status = ', I6 ) " ) i, inform%status
     END IF

     CALL TRB_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( nlp%X, nlp%X_l, nlp%x_u, nlp%G )
   DEALLOCATE( nlp%H%row, nlp%H%col, nlp%H%val, nlp%H%type, userdata%real )
   WRITE( 6, "( /, ' tests completed' )" )

CONTAINS

   SUBROUTINE WHICH_sls( control )
   TYPE ( TRB_control_type ) :: control
#include "galahad_sls_defaults_ls.h"
   control%TRS_control%symmetric_linear_solver = symmetric_linear_solver
   control%TRS_control%definite_linear_solver = definite_linear_solver
   control%PSLS_control%definite_linear_solver = definite_linear_solver
   END SUBROUTINE WHICH_sls

   SUBROUTINE FUN( status, X, userdata, f )     ! Objective function
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), INTENT( OUT ) :: f
   REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( IN ) :: X
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   f = ( X( 1 ) + X( 3 ) + userdata%real( 1 ) ) ** 2 +                         &
       ( X( 2 ) + X( 3 ) ) ** 2 + COS( X( 1 ) )
   status = 0
   RETURN
   END SUBROUTINE FUN

   SUBROUTINE GRAD( status, X, userdata, G )    ! gradient of the objective
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   G( 1 ) = 2.0_rp_ * ( X( 1 ) + X( 3 ) + userdata%real( 1 ) ) - SIN( X( 1 ) )
   G( 2 ) = 2.0_rp_ * ( X( 2 ) + X( 3 ) )
   G( 3 ) = 2.0_rp_ * ( X( 1 ) + X( 3 ) + userdata%real( 1 ) ) +               &
            2.0_rp_ * ( X( 2 ) + X( 3 ) )
   status = 0
   RETURN
   END SUBROUTINE GRAD

   SUBROUTINE HESS( status, X, userdata, Hval ) ! Hessian of the objective
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Hval
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   Hval( 1 ) = 2.0_rp_ - COS( X( 1 ) )
   Hval( 2 ) = 2.0_rp_
   Hval( 3 ) = 2.0_rp_
   Hval( 4 ) = 2.0_rp_
   Hval( 5 ) = 4.0_rp_
   status = 0
   RETURN
   END SUBROUTINE HESS

   SUBROUTINE HESSPROD( status, X, userdata, U, V, got_h ) ! Hessian-vector prod
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: U
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, V
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
   U( 1 ) = U( 1 ) + 2.0_rp_ * ( V( 1 ) + V( 3 ) ) - COS( X( 1 ) ) * V( 1 )
   U( 2 ) = U( 2 ) + 2.0_rp_ * ( V( 2 ) + V( 3 ) )
   U( 3 ) = U( 3 ) + 2.0_rp_ * ( V( 1 ) + V( 2 ) + 2.0_rp_ * V( 3 ) )
   status = 0
   RETURN
   END SUBROUTINE HESSPROD

   SUBROUTINE PREC( status, X, userdata, U, V ) ! apply preconditioner
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: U
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V, X
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   U( 1 ) = 0.5_rp_ * V( 1 )
   U( 2 ) = 0.5_rp_ * V( 2 )
   U( 3 ) = 0.25_rp_ * V( 3 )
   status = 0
   RETURN
   END SUBROUTINE PREC

   SUBROUTINE SHESSPROD( status, X, userdata, nnz_v, INDEX_nz_v, V,            &
                         nnz_u, INDEX_nz_u, U, got_h )
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: nnz_v
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: nnz_u
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( IN ) :: INDEX_nz_v
   INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( OUT ) :: INDEX_nz_u
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: U
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
   INTEGER ( KIND = ip_ ) :: i, j
   REAL ( KIND = rp_ ), DIMENSION( 3 ) :: P
   LOGICAL, DIMENSION( 3 ) :: USED
   P = 0.0_rp_
   USED = .FALSE.
   DO i = 1, nnz_v
     j = INDEX_nz_v( i )
     SELECT CASE( j )
     CASE( 1 )
       P( 1 ) = P( 1 ) + 2.0_rp_ * V( 1 ) - COS( X( 1 ) ) * V( 1 )
       USED( 1 ) = .TRUE.
       P( 3 ) = P( 3 ) + 2.0_rp_ * V( 1 )
       USED( 3 ) = .TRUE.
     CASE( 2 )
       P( 2 ) = P( 2 ) + 2.0_rp_ * V( 2 )
       USED( 2 ) = .TRUE.
       P( 3 ) = P( 3 ) + 2.0_rp_ * V( 2 )
       USED( 3 ) = .TRUE.
     CASE( 3 )
       P( 1 ) = P( 1 ) + 2.0_rp_ * V( 3 )
       USED( 1 ) = .TRUE.
       P( 2 ) = P( 2 ) + 2.0_rp_ * V( 3 )
       USED( 2 ) = .TRUE.
       P( 3 ) = P( 3 ) + 4.0_rp_ * V( 3 )
       USED( 3 ) = .TRUE.
     END SELECT
   END DO
   nnz_u = 0
   DO j = 1, 3
     IF ( USED( j ) ) THEN
       U( j ) = P( j )
       nnz_u = nnz_u + 1
       INDEX_nz_u( nnz_u ) = j
     END IF
   END DO
   status = 0
   RETURN
   END SUBROUTINE SHESSPROD

   END PROGRAM GALAHAD_TRB_test
