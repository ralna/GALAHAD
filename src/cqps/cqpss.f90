! THIS VERSION: GALAHAD 2.4 - 08/12/2009 AT 12:00 GMT.
   PROGRAM GALAHAD_CQPS_EXAMPLE
   USE GALAHAD_CQPS_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( CQPS_data_type ) :: data
   TYPE ( CQPS_control_type ) :: control        
   TYPE ( CQPS_inform_type ) :: inform
   TYPE ( NLPT_userdata_type ) :: userdata
   INTEGER :: s
   INTEGER, PARAMETER :: n = 3, m = 2, h_ne = 4, a_ne = 4 
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_stat, B_stat
! start problem data
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.           ! new structure
   p%n = n ; p%m = m ; p%f = 1.0_wp           ! dimensions & objective constant
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)         ! objective gradient
   p%C_l = (/ 1.0_wp, 2.0_wp /)               ! constraint lower bound
!  p%C_l = (/ 2.0_wp, 2.0_wp /)               ! constraint lower bound
   p%C_u = (/ 2.0_wp, 2.0_wp /)               ! constraint upper bound
   p%X_l = (/ - 1.0_wp, - infinity, - infinity /) ! variable lower bound
   p%X_u = (/ 1.0_wp, infinity, 2.0_wp /)     ! variable upper bound
!  p%X_l = (/ 0.0_wp, 0.0_wp, 0.0_wp /) ! variable lower bound
!  p%X_u = (/ infinity, infinity, infinity /)     ! variable upper bound
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp ! start from zero
!  sparse co-ordinate storage format
   CALL SMT_put( p%H%type, 'COORDINATE', s )     ! Specify co-ordinate 
   CALL SMT_put( p%A%type, 'COORDINATE', s )     ! storage for H and A
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%H%val = (/ 1.0_wp, 2.0_wp, 1.0_wp, 3.0_wp /) ! Hessian H
   p%H%row = (/ 1, 2, 2, 3 /)                     ! NB lower triangle
   p%H%col = (/ 1, 2, 1, 3 /) ; p%H%ne = h_ne
   p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! Jacobian A
   p%A%row = (/ 1, 1, 2, 2 /)
   p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
! problem data complete
   CALL CQPS_initialize( data, control, inform ) ! Initialize control parameters
   control%infinity = infinity                  ! Set infinity
!  control%initial_rho = 10.0_wp
!  control%initial_eta = 10.0_wp
   control%print_level = 1
   control%bqp_control%print_level = 1
   control%bqp_control%maxit = 20
   control%bqp_control%zero_curvature = 1.0D-12
   inform%status = 1
   CALL CQPS_solve( p, C_stat, B_stat, data, control, inform, userdata ) ! Solve
   IF ( inform%status == 0 ) THEN               !  Successful return
     WRITE( 6, "( ' CQPS: ', I0, ' BQP iterations  ', /,                       &
    &     ' Optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%BQP_inform%iter, inform%obj, p%X
   ELSE                                       !  Error returns
     WRITE( 6, "( ' CQPS_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL CQPS_terminate( data, control, inform )  !  delete internal workspace
   END PROGRAM GALAHAD_CQPS_EXAMPLE

