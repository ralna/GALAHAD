   PROGRAM GALAHAD_UGO_EXAMPLE  !  GALAHAD 4.0 - 2022-03-07 AT 10:05 GMT
   USE GALAHAD_UGO_double                       ! double precision version
   USE GALAHAD_USERDATA_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( UGO_control_type ) :: control
   TYPE ( UGO_inform_type ) :: inform
   TYPE ( UGO_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
   EXTERNAL :: FGH
   REAL ( KIND = wp ) :: x_l, x_u, x, f, g, h
   REAL ( KIND = wp ), PARAMETER :: a = 10.0_wp
   x_l = - 1.0_wp; x_u = 2.0_wp                 ! bounds on x
   ALLOCATE( userdata%real( 1 ) )               ! Allocate space for parameter
   userdata%real( 1 ) = a                       ! Record parameter, a
   CALL UGO_initialize( data, control, inform )
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
   USE GALAHAD_USERDATA_double
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), INTENT( IN ) :: x
   REAL ( KIND = wp ), INTENT( OUT ) :: f, g, h
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   REAL ( KIND = wp ) :: a, ax, sax, cax
   a = userdata%real( 1 )
   ax = a * x
   sax = SIN( ax )
   cax = COS( ax )
   f = x * x * cax
   g = - ax * x * sax + 2.0_wp * x * cax
   h = - a * a* x * x * cax - 4.0_wp * ax * sax + 2.0_wp * cax
   status = 0
   RETURN
   END SUBROUTINE FGH
