! THIS VERSION: GALAHAD 5.0 - 2024-06-13 AT 13:50 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_NLS_test_deck2
   USE GALAHAD_USERDATA_precision
   USE GALAHAD_NLS_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( NLS_control_type ) :: control
   TYPE ( NLS_inform_type ) :: inform
   TYPE ( NLS_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: W
   REAL ( KIND = rp_ ), PARAMETER :: p = 1.0_rp_
   REAL ( KIND = rp_ ), PARAMETER :: mult = 1.0_rp_
!  EXTERNAL :: RES, JAC, HESS, JACPROD, HESSPROD, RHESSPRODS, SCALE
   INTEGER ( KIND = ip_ ) :: s, model, scaling, rev, usew

!  ============================================
!  test of scaling, model and weighting options
!  ============================================

! start problem data
   nlp%n = 2 ;  nlp%m = 3 ; nlp%J%ne = 5 ; nlp%H%ne = 2 ; nlp%P%ne = 2
   ALLOCATE( nlp%X( nlp%n ), nlp%C( nlp%m ), nlp%G( nlp%n ), W( nlp%m ) )
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%J%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%J%val( nlp%J%ne ), nlp%J%row( nlp%J%ne ), nlp%J%col( nlp%J%ne))
   nlp%J%row = (/ 1, 2, 2, 3, 3 /)              ! Jacobian J
   nlp%J%col = (/ 1, 1, 2, 1, 2 /)
   ALLOCATE( userdata%real( 1 ) )  ! Allocate space to hold parameter
   userdata%real( 1 ) = p          ! Record parameter, p
   W = 1.0_rp_                      ! Record weight (if needed)
! problem data complete

   WRITE( 6, "( /, ' test of scaling, model & weighting options ', / )" )

   DO model = 1, 8
!  DO model = 3, 3
     IF ( model == 4 ) THEN
     CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
       ALLOCATE( nlp%H%val( nlp%H%ne ), nlp%H%row( nlp%H%ne ),                 &
                 nlp%H%col( nlp%H%ne ) )
       nlp%H%row = (/ 1, 2 /) ! Hessian H
       nlp%H%col = (/ 1, 2 /) ! NB lower triangle only
     ELSE IF ( model == 6 ) THEN
       ALLOCATE( nlp%P%val( nlp%P%ne ), nlp%P%row( nlp%P%ne ),                 &
                 nlp%P%ptr( nlp%m + 1 ) )
       nlp%P%row = (/ 1, 2 /)  ! Hessian products
       nlp%P%ptr = (/ 1, 2, 3, 3 /)
     END IF
if(model /= 3) cycle
     DO scaling = - 1, 8
if(scaling /= 8) cycle
!    DO scaling = 1, 1
!    DO scaling = - 1, - 1
       IF ( scaling == 8 .AND. model == 4 ) CYCLE
!      IF ( scaling == 0 .OR. scaling == 6 ) CYCLE
       DO rev = 0, 1
!      DO rev = 0, 0
if(rev /=0) cycle
         DO usew = 0, 1
!        DO usew = 0, 0
if(usew /=0) cycle
         CALL NLS_initialize( data, control, inform )! Initialize controls

         control%print_level = 1
         control%subproblem_control%print_level = 1
         control%glrt_control%print_level = 1
         control%subproblem_control%glrt_control%print_level = 1
         control%psls_control%print_level = 1

         CALL WHICH_sls( control )
         control%model = model             ! set model
         control%norm = scaling            ! set scaling norm
         control%jacobian_available = 2    ! the Jacobian is available
         IF ( model >= 4 ) control%hessian_available = 2 ! Hessian is available
!        control%print_level = 4
!        control%subproblem_control%print_level = 4
!        control%print_level = 4
!        control%maxit = 1
!        control%subproblem_control%magic_step = .TRUE.
!        control%subproblem_control%glrt_control%print_level = 3
         nlp%X = 1.0_rp_                               ! start from one
         inform%status = 1                            ! set for initial entry
         IF ( rev == 0 ) THEN
           IF ( usew == 0 ) THEN
             CALL NLS_solve( nlp, control, inform, data, userdata,             &
                             eval_C = RES, eval_J = JAC, eval_H = HESS,        &
                             eval_JPROD = JACPROD, eval_HPROD = HESSPROD,      &
                             eval_HPRODS = RHESSPRODS )
           ELSE
             CALL NLS_solve( nlp, control, inform, data, userdata,             &
                             eval_C = RES, eval_J = JAC, eval_H = HESS,        &
                             eval_JPROD = JACPROD, eval_HPROD = HESSPROD,      &
                             eval_HPRODS = RHESSPRODS, W = W )
           END IF
         ELSE
           DO              ! Loop to solve problem
             IF ( usew == 0 ) THEN
               CALL NLS_solve( nlp, control, inform, data, userdata )
             ELSE
               CALL NLS_solve( nlp, control, inform, data, userdata, W = W )
             END IF
             SELECT CASE ( inform%status )   ! reverse communication
             CASE ( 2 )    ! Obtain the residuals
               CALL RES( data%eval_status, nlp%X, userdata, nlp%C )
             CASE ( 3 )    ! Obtain the Jacobian
               CALL JAC( data%eval_status, nlp%X, userdata, nlp%J%val )
             CASE ( 4 )    ! Obtain the Hessian
               CALL HESS( data%eval_status, nlp%X, data%Y, userdata,           &
                          nlp%H%val )
             CASE ( 5 )    ! form a Jacobian-vector product
               CALL JACPROD( data%eval_status, nlp%X, userdata,                &
                             data%transpose, data%U, data%V )
             CASE ( 6 )    ! form a Hessian-vector product
               CALL HESSPROD( data%eval_status, nlp%X, data%Y, userdata,       &
                              data%U, data%V )
             CASE ( 7 )    ! form residual Hessian-vector products
               CALL RHESSPRODS( data%eval_status, nlp%X, data%V, userdata,     &
                                nlp%P%val )
             CASE ( 8 )     ! Apply the preconditioner
               CALL SCALE( data%eval_status, nlp%X, userdata, data%U, data%V )
             CASE DEFAULT   ! Terminal exit from loop
               EXIT
             END SELECT
           END DO
         END IF

         IF ( inform%status == 0 ) THEN
           WRITE( 6, "( I1, ',', I2, 2( ',', I1 ), ':', I6, ' iterations.',    &
          &  ' Optimal objective value = ', F6.1, ' status = ', I6 )" )        &
            rev, scaling, model, usew, inform%iter, inform%norm_c, inform%status
         ELSE
           WRITE( 6, "( I1, ',', I2, 2( ',', I1 ), ': NLS_solve exit status',  &
          & ' = ', I6 )" ) rev, scaling, model, usew, inform%status
         END IF
         CALL NLS_terminate( data, control, inform ) ! delete workspace
       END DO
       END DO
     END DO
   END DO
   DEALLOCATE( nlp%X, nlp%C, nlp%G, W, userdata%real )
   DEALLOCATE( nlp%J%val, nlp%J%row, nlp%J%col, nlp%J%type )
   DEALLOCATE( nlp%H%val, nlp%H%row, nlp%H%col, nlp%H%type )
   DEALLOCATE( nlp%P%val, nlp%P%row, nlp%P%ptr )
   IF ( ALLOCATED( nlp%P%type ) ) DEALLOCATE( nlp%P%type )
   WRITE( 6, "( /, ' test completed' )" )

   CONTAINS

     SUBROUTINE WHICH_sls( control )
     TYPE ( NLS_control_type ) :: control
#include "galahad_sls_defaults.h"
     control%PSLS_control%definite_linear_solver = definite_linear_solver
     control%RQS_control%symmetric_linear_solver = symmetric_linear_solver
     control%RQS_control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls

     SUBROUTINE RES( status, X, userdata, C )
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( OUT ) :: C
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     C( 1 ) = X( 1 ) ** 2 + userdata%real( 1 )
     C( 2 ) = X( 1 ) + X( 2 ) ** 2
     C( 3 ) = X( 1 ) - X( 2 )
     status = 0
     END SUBROUTINE RES

     SUBROUTINE JAC( status, X, userdata, J_val )
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( OUT ) :: J_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     J_val( 1 ) = 2.0_rp_ * X( 1 )
     J_val( 2 ) = 1.0_rp_
     J_val( 3 ) = 2.0_rp_ * X( 2 )
     J_val( 4 ) = 1.0_rp_
     J_val( 5 ) = - 1.0_rp_
     status = 0
     END SUBROUTINE JAC

     SUBROUTINE HESS( status, X, Y, userdata, H_val )
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: H_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     H_val( 1 ) = 2.0_rp_ * Y( 1 )
     H_val( 2 ) = 2.0_rp_ * Y( 2 )
     status = 0
     END SUBROUTINE HESS

     SUBROUTINE JACPROD( status, X, userdata, transpose, U, V, got_j )
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     LOGICAL, INTENT( IN ) :: transpose
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_j
     IF ( transpose ) THEN
       U( 1 ) = U( 1 ) + 2.0_rp_ * X( 1 ) * V( 1 ) + V( 2 ) + V( 3 )
       U( 2 ) = U( 2 ) + 2.0_rp_ * X( 2 ) * V( 2 ) - V( 3 )
     ELSE
       U( 1 ) = U( 1 ) + 2.0_rp_ * X( 1 ) * V( 1 )
       U( 2 ) = U( 2 ) + V( 1 )  + 2.0_rp_ * X( 2 ) * V( 2 )
       U( 3 ) = U( 3 ) + V( 1 ) - V( 2 )
     END IF
     status = 0
     END SUBROUTINE JACPROD

     SUBROUTINE HESSPROD( status, X, Y, userdata, U, V, got_h )
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
     U( 1 ) = U( 1 ) + 2.0_rp_ * Y( 1 ) * V( 1 )
     U( 2 ) = U( 2 ) + 2.0_rp_ * Y( 2 ) * V( 2 )
     status = 0
     END SUBROUTINE HESSPROD

     SUBROUTINE RHESSPRODS( status, X, V, userdata, P_val, got_h )
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: P_val
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
     P_val( 1 ) = 2.0_rp_ * V( 1 )
     P_val( 2 ) = 2.0_rp_ * V( 2 )
     status = 0
     END SUBROUTINE RHESSPRODS

     SUBROUTINE SCALE( status, X, userdata, U, V )
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, V
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: U
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
!     U( 1 ) = 0.5_rp_ * V( 1 )
!     U( 2 ) = 0.5_rp_ * V( 2 )
     U( 1 ) = V( 1 )
     U( 2 ) = V( 2 )
     status = 0
     END SUBROUTINE SCALE

   END PROGRAM GALAHAD_NLS_test_deck2
