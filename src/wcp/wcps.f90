! THIS VERSION: GALAHAD 2.4 - 04/05/2010 AT 09:00 GMT.
   PROGRAM GALAHAD_WCP_example
   USE GALAHAD_WCP_double                    ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( WCP_data_type ) :: data
   TYPE ( WCP_control_type ) :: control        
   TYPE ( WCP_inform_type ) :: inform
   INTEGER, PARAMETER :: n = 3, m = 2, a_ne = 4 
   INTEGER :: i, s
! start problem data
   ALLOCATE( p%X( n ), p%X_l( n ), p%X_u( n ), p%Z_l( n ), p%Z_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ), p%Y_l( m ), p%Y_u( m ) )
   p%new_problem_structure = .TRUE.           ! new structure
   p%n = n ; p%m = m ; p%f = 0.0_wp           ! dimensions & objective constant
   p%C_l = (/ 1.0_wp, 2.0_wp /)               ! constraint lower bound
   p%C_u = (/ 2.0_wp, 2.0_wp /)               ! constraint upper bound
   p%X_l = (/ - 1.0_wp, - infinity, - infinity /) ! variable lower bound
   p%X_u = (/ 1.0_wp, infinity, 2.0_wp /)     ! variable upper bound
   p%gradient_kind = 0   
! sparse co-ordinate storage format: integer components
   CALL SMT_put( p%A%type, 'COORDINATE', s )     ! storage for A
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%A%row = (/ 1, 1, 2, 2 /)                 ! Jacobian A
   p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
! integer components complete   
   CALL WCP_initialize( data, control, inform ) ! Initialize control parameters
   control%infinity = infinity                  ! Set infinity
   p%X = (/  -2.0_wp, 1.0_wp,  3.0_wp /)        ! set x0
   p%Y_l = 1.0_wp ;  p%Y_u = - 1.0_wp ; p%Z_l = 1.0_wp ;  p%Z_u = - 1.0_wp
! sparse co-ordinate storage format: real components
   p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! Jacobian A
! real components complete   
   CALL WCP_solve( p, data, control, inform )   ! Solve problem
   IF ( inform%status == 0 ) THEN               ! Successful return
     WRITE( 6, "( 1X, I0, ' iterations. objective value =', ES11.4, /,         &
    &  ' well-centered point:', /, ' i     X_l           X         X_u' )" )   &
       inform%iter, inform%obj
     DO i = 1, n
       WRITE( 6, "( I2, 3ES12.4 )" ) i, p%X_l( i ), p%X( i ), p%X_u( i )
     END DO
     WRITE( 6, "( ' constraints:', /, ' i     C_l       A * X         C_u' )" )
     DO i = 1, m
       WRITE( 6, "( I2, 3ES12.4 )" ) i, p%C_l( i ), p%C( i ), p%C_u( i )
     END DO
   ELSE                                        !  Error returns
     WRITE( 6, "( ' WCP_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL WCP_terminate( data, control, inform ) !  delete internal workspace
   END PROGRAM GALAHAD_WCP_example

