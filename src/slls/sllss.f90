! THIS VERSION: GALAHAD 5.5 - 2026-02-01 AT 08:40 GMT
   PROGRAM GALAHAD_SLLS_EXAMPLE
   USE GALAHAD_SLLS_double                   ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( QPT_problem_type ) :: p
   TYPE ( SLLS_data_type ) :: data
   TYPE ( SLLS_control_type ) :: control
   TYPE ( SLLS_inform_type ) :: inform
   TYPE ( USERDATA_type ) :: userdata
   INTEGER :: s
   INTEGER, PARAMETER :: n = 3, o = 4, m = 1, ao_ne = 5
! start problem data
   ALLOCATE( p%B( o ), p%X( n ), p%X_status( n ) )
   p%n = n ; p%o = o ; p%m = m                ! dimensions
   p%B = (/ 0.0_wp, 2.0_wp, 1.0_wp, 2.0_wp /) ! right-hand side
   p%X = 0.0_wp ! start from zero
!  sparse co-ordinate storage format
   CALL SMT_put( p%Ao%type, 'COORDINATE', s )  ! Co-ordinate storage for Ao
   ALLOCATE( p%Ao%val( ao_ne ), p%Ao%row( ao_ne ), p%Ao%col( ao_ne ) )
   p%Ao%m = o ; p%Ao%n = n ; p%Ao%ne = ao_ne  ! design matrix Ao
   p%Ao%row = (/ 1, 2, 2, 3, 4 /) ; p%Ao%col = (/ 1, 1, 2, 3, 3 /)
   p%Ao%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
! problem data complete
   CALL SLLS_initialize( data, control, inform ) ! Initialize control parameters
   control%SBLS_control%symmetric_linear_solver = 'sytr' ! non-default solver
   control%SBLS_control%definite_linear_solver = 'potr' ! non-default solver
!  control%print_level = 1                       ! print one line/iteration
   inform%status = 1
   CALL SLLS_solve( p, data, control, inform, userdata )
   IF ( inform%status == 0 ) THEN             !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations  ', /,                        &
    &     ' Optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
       inform%iter, inform%obj, p%X
     WRITE( 6, "( ' Lagrange multiplier estimate =', ES12.4 )" ) p%Y
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve exit status = ', I0 ) " ) inform%status
     WRITE( 6, * ) inform%alloc_status, inform%bad_alloc
   END IF
   CALL SLLS_terminate( data, control, inform )  !  delete workspace
   DEALLOCATE( p%B, p%X, p%Y, p%Z, p%R, p%G, p%X_status )
   DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col, p%Ao%type )
   END PROGRAM GALAHAD_SLLS_EXAMPLE
