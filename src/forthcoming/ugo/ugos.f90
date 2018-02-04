   PROGRAM GALAHAD_UGO_EXAMPLE  !  GALAHAD 2.8 - 03/06/2016 AT 08:35 GMT
   USE GALAHAD_UGO_double                       ! double precision version
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( UGO_control_type ) :: control
   TYPE ( UGO_inform_type ) :: inform
   TYPE ( UGO_data_type ) :: data
   TYPE ( NLPT_userdata_type ) :: userdata
   EXTERNAL :: FGH
   REAL ( KIND = wp ) :: x_l, x_u, x, f, g, h
!  REAL ( KIND = wp ), PARAMETER :: p = 4.0_wp
   REAL ( KIND = wp ), PARAMETER :: p = 0.0_wp
   x_l = - 1.0_wp; x_u = 2.0_wp                 ! bounds on x
   ALLOCATE( userdata%real( 1 ) )               ! Allocate space for parameter
   userdata%real( 1 ) = p                       ! Record parameter, p
   CALL UGO_initialize( data, control, inform )
   control%print_level = 1
   control%maxit = 100
   control%lipschitz_estimate_used = 3
   inform%status = 1                            ! set for initial entry
   CALL UGO_solve( x_l, x_u, x, f, g, h, control, inform, data, userdata,      &
                   eval_FGH = FGH )   ! Solve problem
   IF ( inform%status == 0 ) THEN               ! Successful return
     WRITE( 6, "( ' UGO: ', I0, ' evaluations', /,                             &
    &     ' Optimal solution =', ES14.6, /,                                    &
    &     ' Optimal objective value and gradient =', 2ES14.6 )" )              &
              inform%iter, x, f, g
   ELSE                                         ! Error returns
     WRITE( 6, "( ' UGO_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL UGO_terminate( data, control, inform )  ! delete internal workspace
   END PROGRAM GALAHAD_UGO_EXAMPLE

   SUBROUTINE FGH( status, x, userdata, f, g, h )
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), INTENT( IN ) :: x
   REAL ( KIND = wp ), INTENT( OUT ) :: f, g, h
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   REAL ( KIND = wp ) :: a
!  f = - 0.5_wp * x ** 2 + userdata%real( 1 ) * x
!  g = - x + userdata%real( 1 )
!  h = - 1.0_wp
   a = 10.0_wp
   f = x * x * cos( a*x )
   g = - a * x * x * sin( a*x ) + 2.0_wp * x * cos( a*x )
   h = - a * a* x * x * cos( a*x ) - 4.0_wp * a * x * sin( a*x ) + 2.0_wp * cos( a*x )
   status = 0
   RETURN
   END SUBROUTINE FGH
