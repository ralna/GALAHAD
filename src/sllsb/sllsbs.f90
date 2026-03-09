! THIS VERSION: GALAHAD 5.5 - 2026-01-19 AT 10:30 GMT.
   PROGRAM GALAHAD_SLLSB_EXAMPLE
   USE GALAHAD_SLLSB_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( QPT_problem_type ) :: p
   TYPE ( SLLSB_data_type ) :: data
   TYPE ( SLLSB_control_type ) :: control
   TYPE ( SLLSB_inform_type ) :: inform
   INTEGER :: s
   INTEGER, PARAMETER :: n = 3, o = 4, m = 1, ao_ne = 5
! start problem data
   ALLOCATE( p%B( o ), p%X( n ), p%Y( m ), p%Z( n ) )
   p%n = n ; p%o = o ; p%m = m ! dimensions
   p%B = (/ 0.0_wp, 2.0_wp, 1.0_wp, 2.0_wp /) ! observations
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp
 ! start from zero
!  sparse co-ordinate storage format
   CALL SMT_put( p%Ao%type, 'COORDINATE', s )  ! Co-ordinate  storage for Ao
   ALLOCATE( p%Ao%val( ao_ne ), p%Ao%row( ao_ne ), p%Ao%col( ao_ne ) )
   p%Ao%m = o ; p%Ao%n = n
   p%Ao%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! design matrix Ao
   p%Ao%row = (/ 1, 2, 2, 3, 4 /)
   p%Ao%col = (/ 1, 1, 2, 3, 3 /) ; p%Ao%ne = ao_ne
! problem data complete
   CALL SLLSB_initialize( data, control, inform ) ! Initialize control params
   control%symmetric_linear_solver = 'sytr '
   control%FDC_control%symmetric_linear_solver = 'sytr '
!  control%print_level = 1                       ! print one line/iteration
   inform%status = 1
   CALL SLLSB_solve( p, data, control, inform )
   IF ( inform%status == 0 ) THEN             !  Successful return
     WRITE( 6, "( /, ' SLLSB: ', I0, ' iterations  ', /,                       &
    &     ' Optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
       inform%iter, inform%obj, p%X
     WRITE( 6, "( ' Lagrange multiplier estimate =', ES12.4 )" ) p%Y
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLSB_solve exit status = ', I0 ) " ) inform%status
     WRITE( 6, * ) inform%alloc_status, inform%bad_alloc
   END IF
   CALL SLLSB_terminate( data, control, inform )  !  delete workspace
   DEALLOCATE( p%B, p%X, p%Y, p%Z, p%R, p%X_status )
   DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col, p%Ao%type )
   END PROGRAM GALAHAD_SLLSB_EXAMPLE
