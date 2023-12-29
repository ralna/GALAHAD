! THIS VERSION: GALAHAD 4.3 - 2023-12-27 AT 13:10 GMT.
   PROGRAM GALAHAD_BLLSB_EXAMPLE
   USE GALAHAD_BLLSB_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( BLLSB_data_type ) :: data
   TYPE ( BLLSB_control_type ) :: control
   TYPE ( BLLSB_inform_type ) :: inform
   INTEGER :: s
   INTEGER, PARAMETER :: n = 3, o = 4, ao_ne = 7
   REAL ( KIND = wp ), DIMENSION( o ) :: W
! start problem data
   ALLOCATE( p%B( o ), p%R( o ) )
   ALLOCATE( p%X_l( n ), p%X_u( n ), p%X( n ), p%Z( n ), p%X_status( n ) )
   p%new_problem_structure = .TRUE.           ! new structure
   p%n = n ; p%o = o                          ! dimensions
   p%B = (/ 2.0_wp, 2.0_wp, 3.0_wp, 1.0_wp /) ! observations
   p%X_l = (/ - 1.0_wp, - infinity, - infinity /) ! variable lower bound
   p%X_u = (/ 1.0_wp, infinity, 2.0_wp /)     ! variable upper bound
   p%X = 0.0_wp ; p%Z = 0.0_wp                ! start from zero
   W = (/ 1.0_wp, 1.0_wp, 1.0_wp, 2.0_wp /)   ! weights
!  sparse co-ordinate storage format
   CALL SMT_put( p%Ao%type, 'COORDINATE', s ) ! Specify co-ordinate
   ALLOCATE( p%Ao%row( ao_ne ), p%Ao%col( ao_ne ), p%Ao%val( ao_ne ) )
   p%Ao%row = (/ 1, 1, 2, 2, 3, 3, 4 /) ; p%Ao%ne = ao_ne  ! Ao
   p%Ao%col = (/ 1, 2, 2, 3, 1, 3, 2 /)
   p%Ao%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
! problem data complete
   CALL BLLSB_initialize( data, control, inform ) ! Initialize control params
   control%symmetric_linear_solver = 'sytr '
   control%FDC_control%symmetric_linear_solver = 'sytr '
   control%infinity = infinity                        ! Set infinity
   CALL BLLSB_solve( p, data, control, inform, W = W ) ! Solve
   IF ( inform%status == 0 ) THEN                     ! Successful return
     WRITE( 6, "( ' BLLSB: ', I0, ' iterations  ', /,                          &
    &     ' Optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%iter, inform%obj, p%X
   ELSE                                         ! Error returns
     WRITE( 6, "( ' BLLSB_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL BLLSB_terminate( data, control, inform )  !  delete internal workspace
   END PROGRAM GALAHAD_BLLSB_EXAMPLE
