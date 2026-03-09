! THIS VERSION: GALAHAD 5.5 - 2026-02-01 AT 10:10 GMT
   PROGRAM GALAHAD_SLLSB_MULTIPLE_EXAMPLE
   USE GALAHAD_SLLSB_double ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( QPT_problem_type ) :: p
   TYPE ( SLLSB_data_type ) :: data
   TYPE ( SLLSB_control_type ) :: control
   TYPE ( SLLSB_inform_type ) :: inform
   INTEGER :: j, m, s
   INTEGER, PARAMETER :: n = 5, o = n + 1, a_ne = 2 * n + 2
! set problem data
   ALLOCATE( p%B( o ), p%X( n ), p%X_status( n ), p%COHORT( n ) )
   p%COHORT = [ 1, 2, 0, 1, 2 ] ! cohorts
   m = MAXVAL( p%COHORT ) ! number of cohorts
   p%n = n ; p%o = o ; p%m = m ! dimensions
   p%B = [ 0.0_wp, 2.0_wp, 1.0_wp, 2.0_wp, 1.0_wp, 2.0_wp ] ! observations
   p%X = 0.0_wp ! start from zero
!  sparse co-ordinate storage format for the design matrix Ao
   CALL SMT_put( p%Ao%type, 'COORDINATE', s ) ! Co-ordinate  storage for Ao
   ALLOCATE( p%Ao%val( a_ne ), p%Ao%row( a_ne ), p%Ao%col( A_ne ) )
   p%Ao%m = o ; p%Ao%n = n ; p%Ao%ne = 0
   DO j = 1, n
     p%Ao%ne = p%Ao%ne + 1
     p%Ao%row( p%Ao%ne ) = j
     p%Ao%col( p%Ao%ne ) = j
     p%Ao%val( p%Ao%ne ) = 1.0_wp
     p%Ao%ne = p%Ao%ne + 1
     p%Ao%row( p%Ao%ne ) = j + 1
     p%Ao%col( p%Ao%ne ) = j
     p%Ao%val( p%Ao%ne ) = - 1.0_wp
   END DO
! problem data complete
   CALL SLLSB_initialize( data, control, inform ) ! Initialize control params
!  control%print_level = 1 ! print one line/iteration
   control%maxit = 20      ! maximum of 20 iterations
   control%symmetric_linear_solver = 'sytr '
   control%FDC_control%symmetric_linear_solver = 'sytr '
   inform%status = 1
   CALL SLLSB_solve( p, data, control, inform )
   IF ( inform%status == 0 ) THEN  !  Successful return
     WRITE( 6, "( /, ' SLLSB: ', I0, ' iterations  ', /,                       &
    &     ' Optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution =', ( 5ES12.4 ) )" )                 &
       inform%iter, inform%obj, p%X
     WRITE( 6, "( ' Lagrange multiplier estimates =', ( 5ES12.4 ) )" ) p%Y
   ELSE                            ! Error returns
     WRITE( 6, "( /, ' SLLSB_solve exit status = ', I0 ) " ) inform%status
     WRITE( 6, "( /, ' objective value =',                                     &
    &       ES12.4, /, ' Current solution =', ( 5ES12.4 ) )" ) inform%obj, p%X
   END IF
   CALL SLLSB_terminate( data, control, inform )  !  delete workspace
   DEALLOCATE( p%B, p%X, p%Y, p%Z, p%R, p%X_status, p%COHORT )
   DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col, p%Ao%type )
   END PROGRAM GALAHAD_SLLSB_MULTIPLE_EXAMPLE
