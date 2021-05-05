   PROGRAM GALAHAD_NLS_EXAMPLE2 !  GALAHAD 3.3 - 05/05/2021 AT 14:15 GMT
   USE GALAHAD_NLS_double                     ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( NLS_control_type ) :: control
   TYPE ( NLS_inform_type ) :: inform
   TYPE ( NLS_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
   EXTERNAL :: EVALC, EVALJ, EVALHPROD
   INTEGER :: s
   INTEGER, PARAMETER :: m = 2, n = 3, j_ne = 4, h_ne = 3, p_ne = 3
   REAL ( KIND = wp ), PARAMETER :: p = 4.0_wp  ! parameter p
! start problem data
   nlp%n = n ; nlp%m = m ; nlp%J%ne = j_ne ; nlp%H%ne = h_ne   ! dimensions
   ALLOCATE( nlp%X( n ), nlp%C( m ) )
   nlp%X = (/ 1.0_wp, 1.0_wp, 1.0_wp /)         ! start from (-1,1,1)
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%J%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%J%val( j_ne ), nlp%J%row( j_ne ), nlp%J%col( j_ne ) )
   nlp%J%row = (/ 1, 2, 1, 2 /)                 ! Jacobian J(x)
   nlp%J%col = (/ 1, 2, 3, 3 /)
   ALLOCATE( userdata%real( 1 ) )               ! Allocate space for parameter
   userdata%real( 1 ) = p                       ! Record parameter, p
! problem data complete ; solve using a Newton model
   CALL NLS_initialize( data, control, inform ) ! Initialize control params
   control%jacobian_available = 2               ! Jacobian is available
   control%hessian_available = 1                ! only Hessian-vector products
   control%model = 4                            ! use the Newton model
   inform%status = 1                            ! set for initial entry
   CALL NLS_solve( nlp, control, inform, data, userdata, eval_C = EVALC,       &
              eval_J = EVALJ, eval_HPROD = EVALHPROD )  ! Solve problem
   IF ( inform%status == 0 ) THEN               ! Successful return
     WRITE( 6, "( ' NLS: ', I0, ' iterations -',                               &
    &     ' optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%iter, inform%obj, nlp%X
   ELSE                                         ! Error returns
     WRITE( 6, "( ' NLS_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL NLS_terminate( data, control, inform )  ! delete internal workspace
   DEALLOCATE( nlp%X, nlp%G, nlp%J%val, nlp%J%row, nlp%J%col, userdata%real )
   END PROGRAM GALAHAD_NLS_EXAMPLE2

   SUBROUTINE EVALC( status, X, userdata, C )   ! residual
   USE GALAHAD_USERDATA_double
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ),INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ),INTENT( OUT ) :: C
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   REAL ( KIND = wp ) :: p
   p = userdata%real( 1 )
   C( 1 ) = X( 3 ) * X( 1 ) ** 2 + P
   C( 2 ) = X( 2 ) ** 2 + X( 3 )
   status = 0
   RETURN
   END SUBROUTINE EVALC

   SUBROUTINE EVALJ( status, X, userdata, J_val )    ! Jacobian
   USE GALAHAD_USERDATA_double
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: J_val
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   REAL ( KIND = wp ) :: p
   p = userdata%real( 1 )
   J_val( 1 ) = 2.0_wp * X( 1 ) * X( 3 )
   J_val( 2 ) = 2.0_wp * X( 2 )
   J_val( 3 ) = X( 1 ) ** 2
   J_val( 4 ) = 1.0_wp
   status = 0
   RETURN
   END SUBROUTINE EVALJ

   SUBROUTINE EVALHPROD( status, X, Y, userdata, U, V, got_h ) ! Hessian product
   USE GALAHAD_USERDATA_double
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
   U( 1 ) = U( 1 ) + 2.0_wp * Y( 1 ) * ( X( 3 ) * V( 1 ) + X( 1 ) * V( 3 ) )
   U( 2 ) = U( 2 ) + 2.0_wp * Y( 2 ) * V( 2 )
   U( 3 ) = U( 3 ) + 2.0_wp * Y( 1 ) * X( 1 ) * V( 1 )
   status = 0
   RETURN
   END SUBROUTINE EVALHPROD
