! THIS VERSION: GALAHAD 5.0 - 2024-06-06 AT 16:50 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_DGO_TEST  !! far from complete
   USE GALAHAD_USERDATA_precision
   USE GALAHAD_DGO_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( DGO_control_type ) :: control
   TYPE ( DGO_inform_type ) :: inform
   TYPE ( DGO_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
   EXTERNAL :: FUN, GRAD, HESS, HPROD
   INTEGER ( KIND = ip_ ) :: s, scratch_out
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, h_ne = 5
   REAL ( KIND = rp_ ), PARAMETER :: p = 4.0_rp_
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20    ! infinity
! start problem data
   nlp%pname = 'DGOSPEC'                        ! name
   nlp%n = n ; nlp%H%ne = h_ne                  ! dimensions
   ALLOCATE( nlp%X( n ), nlp%G( n ), nlp%X_l( n ), nlp%X_u( n ) )
!  nlp%X_l = -10.0_rp_ ; nlp%X_u = 0.5_rp_ ! search in [-10,1/2]
   nlp%X_l = -1.0_rp_ ; nlp%X_u = 0.5_rp_ ! search in [-10,1/2]
   nlp%X = 0.0_rp_  ! start from 1.0
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%H%val( h_ne ), nlp%H%row( h_ne ), nlp%H%col( h_ne ) )
   nlp%H%row = (/ 1, 2, 3, 3, 3 /)              ! Hessian H
   nlp%H%col = (/ 1, 2, 1, 2, 3 /)              ! NB lower triangle
   ALLOCATE( userdata%real( 1 ) )               ! Allocate space for parameter
   userdata%real( 1 ) = p                       ! Record parameter, p
! problem data complete
   CALL DGO_initialize( data, control, inform ) ! Initialize control parameters
   CALL WHICH_sls( control )
   OPEN( NEWUNIT = scratch_out, STATUS = 'SCRATCH' )
   control%out = scratch_out ; control%error = scratch_out
   control%print_level = 101
!  control%attempts_max = 10000
   control%maxit = 1000
   control%max_evals = 2000
! Solve the problem
   inform%status = 1                            ! set for initial entry
   CALL DGO_solve( nlp, control, inform, data, userdata, eval_F = FUN,         &
                   eval_G = GRAD, eval_H = HESS, eval_HPROD = HPROD )
   IF ( inform%status == GALAHAD_ok .OR.                                       &
        inform%status == GALAHAD_error_max_iterations .OR.                     &
        inform%status == GALAHAD_error_max_evaluations ) THEN  ! Success
     WRITE( 6, "( ' DGO: ', I0, ' iterations, ', I0, ' evaluations -',         &
    &     ' best objective value =',                                           &
    &       ES12.4, /, ' Best solution = ', ( 5ES12.4 ) )" )                   &
     inform%iter, inform%f_eval, inform%obj, nlp%X
   ELSE                                         ! Error returns
     WRITE( 6, "( ' DGO_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL DGO_terminate( data, control, inform )  ! delete internal workspace
   DEALLOCATE( nlp%X, nlp%G, nlp%X_l, nlp%X_u )
   DEALLOCATE( nlp%H%row, nlp%H%col, nlp%H%val, nlp%H%type, userdata%real )
   CLOSE( UNIT = scratch_out )
   WRITE( 6, "( /, ' tests completed' )" )

CONTAINS

   SUBROUTINE WHICH_sls( control )
   TYPE ( DGO_control_type ) :: control
#include "galahad_sls_defaults_ls.h"
   control%TRB_control%TRS_control%symmetric_linear_solver                     &
     = symmetric_linear_solver
   control%TRB_control%TRS_control%definite_linear_solver                      &
     = definite_linear_solver
   control%TRB_control%PSLS_control%definite_linear_solver                     &
     = definite_linear_solver
   END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_DGO_TEST

   SUBROUTINE FUN( status, X, userdata, f )     ! Objective function
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), INTENT( OUT ) :: f
   REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( IN ) :: X
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   REAL, PARAMETER :: freq = 10.0_rp_
   REAL, PARAMETER :: mag = 1000.0_rp_
   f = ( X( 1 ) + X( 3 ) + userdata%real( 1 ) ) ** 2 +                         &
       ( X( 2 ) + X( 3 ) ) ** 2 + mag * COS( freq * X( 1 ) ) +                 &
         X( 1 ) + X( 2 ) + X( 3 )

   status = 0
   RETURN
   END SUBROUTINE FUN

   SUBROUTINE GRAD( status, X, userdata, G )    ! gradient of the objective
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   REAL, PARAMETER :: freq = 10.0_rp_
   REAL, PARAMETER :: mag = 1000.0_rp_
   G( 1 ) = 2.0_rp_ * ( X( 1 ) + X( 3 ) + userdata%real( 1 ) )                 &
            - mag * freq * SIN( freq * X( 1 ) ) + 1.0_rp_
   G( 2 ) = 2.0_rp_ * ( X( 2 ) + X( 3 ) ) + 1.0_rp_
   G( 3 ) = 2.0_rp_ * ( X( 1 ) + X( 3 ) + userdata%real( 1 ) ) +               &
            2.0_rp_ * ( X( 2 ) + X( 3 ) ) + 1.0_rp_
   status = 0
   RETURN
   END SUBROUTINE GRAD

   SUBROUTINE HESS( status, X, userdata, Hval ) ! Hessian of the objective
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Hval
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   REAL, PARAMETER :: freq = 10.0_rp_
   REAL, PARAMETER :: mag = 1000.0_rp_
   Hval( 1 ) = 2.0_rp_ - mag * freq * freq * COS( freq * X( 1 ) )
   Hval( 2 ) = 2.0_rp_
   Hval( 3 ) = 2.0_rp_
   Hval( 4 ) = 2.0_rp_
   Hval( 5 ) = 4.0_rp_
   status = 0
   RETURN
   END SUBROUTINE HESS

   SUBROUTINE HPROD( status, X, userdata, U, V, got_h ) ! Hessian-vector product
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: U
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
   REAL, PARAMETER :: freq = 10.0_rp_
   REAL, PARAMETER :: mag = 1000.0_rp_
   U( 1 ) = U( 1 )                                                             &
            + ( 2.0_rp_ - mag * freq * freq * COS( freq * X( 1 ) ) ) * V( 1 )  &
            + 2.0_rp_ * V( 3 )
   U( 2 ) = U( 2 ) + 2.0_rp_ * V( 2 ) + 2.0_rp_ * V( 3 )
   U( 3 ) = U( 3 ) + 2.0_rp_ * V( 1 ) + 2.0_rp_ * V( 2 ) + 4.0_rp_ * V( 3 )
   status = 0
   RETURN
   END SUBROUTINE HPROD
