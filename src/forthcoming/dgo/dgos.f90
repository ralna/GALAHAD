   PROGRAM GALAHAD_DGO_EXAMPLE  !  GALAHAD 3.3 - 10/07/2021 AT 09:45 GMT.
   USE GALAHAD_DGO_double                       ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( DGO_control_type ) :: control
   TYPE ( DGO_inform_type ) :: inform
   TYPE ( DGO_data_type ) :: data
   TYPE ( NLPT_userdata_type ) :: userdata
   EXTERNAL :: FUN, GRAD, HESS, HPROD
   INTEGER :: s
   INTEGER, PARAMETER :: n = 2, h_ne = 3
   REAL ( KIND = wp ), PARAMETER :: p = - 2.1_wp
! start problem data
   nlp%pname = 'CAMEL6'                         ! name
   nlp%n = n ; nlp%H%ne = h_ne                  ! dimensions
   ALLOCATE( nlp%X( n ), nlp%G( n ), nlp%X_l( n ), nlp%X_u( n ) )
   nlp%X_l( : n )  = (/ - 3.0_wp, - 2.0_wp /)
   nlp%X_u( : n )  = (/ 3.0_wp, 2.0_wp /)
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%H%val( h_ne ), nlp%H%row( h_ne ), nlp%H%col( h_ne ) )
   nlp%H%row = (/ 1, 2, 2 /)              ! Hessian H
   nlp%H%col = (/ 1, 1, 2 /)              ! NB lower triangle
! problem data complete
   ALLOCATE( userdata%real( 1 ) )               ! Allocate space for parameter
   userdata%real( 1 ) = p                       ! Record parameter, p
   CALL DGO_initialize( data, control, inform ) ! Initialize control parameters
   control%maxit = 2000
!  control%print_level = 1
! Solve the problem
   inform%status = 1                            ! set for initial entry
   CALL DGO_solve( nlp, control, inform, data, userdata, eval_F = FUN,         &
                   eval_G = GRAD, eval_H = HESS, eval_HPROD = HPROD )
   IF ( inform%status == 0 ) THEN               ! Successful return
     WRITE( 6, "( ' DGO: ', I0, ' iterations -',                               &
    &     ' optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%iter, inform%obj, nlp%X
   ELSE                                         ! Error returns
     WRITE( 6, "( ' DGO_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL DGO_terminate( data, control, inform )  ! delete internal workspace
   DEALLOCATE( nlp%X, nlp%G, nlp%H%val, nlp%H%row, nlp%H%col, userdata%real )
   END PROGRAM GALAHAD_DGO_EXAMPLE

   SUBROUTINE FUN( status, X, userdata, f )     ! Objective function
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), INTENT( OUT ) :: f
   REAL ( KIND = wp ), DIMENSION( : ),INTENT( IN ) :: X
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   REAL ( KIND = wp ) :: x1, x2, p
   x1 = X( 1 ) ; x2 = X( 2 ) ; p = userdata%real( 1 )
   f = ( 4.0_wp + p * x1 ** 2 + x1 ** 4 / 3.0_wp ) * x1 ** 2 + x1 * x2 +       &
       ( - 4.0_wp + 4.0_wp * x2 ** 2 ) * x2 ** 2
   status = 0
   RETURN
   END SUBROUTINE FUN

   SUBROUTINE GRAD( status, X, userdata, G )    ! gradient of the objective
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   REAL ( KIND = wp ) :: x1, x2, p
   x1 = X( 1 ) ; x2 = X( 2 ) ; p = userdata%real( 1 )
   G( 1 ) = ( 8.0_wp + 4.0_wp * p * x1 ** 2 + 2.0_wp * x1 ** 4 ) * x1 + x2
   G( 2 ) = x1 + ( - 8.0_wp + 16.0_wp * x2 ** 2 ) * x2
   status = 0
   RETURN
   END SUBROUTINE GRAD

   SUBROUTINE HESS( status, X, userdata, Hval ) ! Hessian of the objective
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: Hval
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   REAL ( KIND = wp ) :: x1, x2, p
   x1 = X( 1 ) ; x2 = X( 2 ) ; p = userdata%real( 1 )
   Hval( 1 ) = 8.0_wp + 12.0_wp * p * x1 ** 2 + 10.0_wp * x1 ** 4
   Hval( 2 ) = 1.0_wp
   Hval( 3 ) = - 8.0_wp + 48.0_wp * x2 * x2
   status = 0
   RETURN
   END SUBROUTINE HESS

   SUBROUTINE HPROD( status, X, userdata, U, V, got_h ) ! Hessian-vector product
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
   REAL ( KIND = wp ) :: x1, x2, p
   x1 = X( 1 ) ; x2 = X( 2 ) ; p = userdata%real( 1 )
   U( 1 ) = U( 1 ) + ( 8.0_wp + 12.0_wp * p * x1 ** 2 + 10.0_wp * x1 ** 4 )    &
              * V( 1 ) +  V( 2 )
   U( 2 ) = U( 2 ) + V( 1 ) + ( - 8.0_wp + 48.0_wp * x2 * x2 ) * V( 2 )
   status = 0
   RETURN
   END SUBROUTINE HPROD
