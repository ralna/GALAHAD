   PROGRAM GALAHAD_UGO_EXAMPLE2  !  GALAHAD 4.0 - 2022-03-07 AT 10:15 GMT
   USE GALAHAD_UGO_double                    ! double precision version
   USE GALAHAD_USERDATA_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( UGO_control_type ) :: control
   TYPE ( UGO_inform_type ) :: inform
   TYPE ( UGO_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
   REAL ( KIND = wp ) :: x_l, x_u, x, f, g, h
   REAL ( KIND = wp ), PARAMETER :: a = 10.0_wp
   REAL ( KIND = wp ) :: ax, sax, cax
   x_l = - 1.0_wp; x_u = 2.0_wp             ! bounds on x
   ALLOCATE( userdata%real( 1 ) )           ! Allocate space for parameter
   userdata%real( 1 ) = a                   ! Record parameter, a
   CALL UGO_initialize( data, control, inform )
   inform%status = 1                        ! set for initial entry
   DO ! Solve problem using reverse communication
     CALL UGO_solve( x_l, x_u, x, f, g, h, control, inform, data, userdata )
     SELECT CASE ( inform%status )
     CASE ( 0 )                             ! Successful return
       WRITE( 6, "( ' UGO: ', I0, ' evaluations', /,                           &
      &     ' Optimal solution =', ES14.6, /,                                  &
      &     ' Optimal objective value and gradient =', 2ES14.6 )" )            &
                inform%iter, x, f, g
       EXIT
     CASE ( 4 )                             ! obtain function/derivative values
       ax = a * x
       sax = SIN( ax )
       cax = COS( ax )
       f = x * x * cax
       g = - ax * x * sax + 2.0_wp * x * cax
       h = - a * a* x * x * cax - 4.0_wp * ax * sax + 2.0_wp * cax
       data%eval_status = 0
     CASE DEFAULT                           ! Error returns
       WRITE( 6, "( ' UGO_solve exit status = ', I6 ) " ) inform%status
       EXIT
     END SELECT
   END DO
   CALL UGO_terminate( data, control, inform )  ! delete internal workspace
   END PROGRAM GALAHAD_UGO_EXAMPLE2
