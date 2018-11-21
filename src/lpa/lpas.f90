! THIS VERSION: GALAHAD 3.1 - 07/10/2018 AT 12:05 GMT.
   PROGRAM GALAHAD_LPA_EXAMPLE
   USE GALAHAD_LPA_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( LPA_data_type ) :: data
   TYPE ( LPA_control_type ) :: control
   TYPE ( LPA_inform_type ) :: inform
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_stat, X_stat
   INTEGER :: s
   INTEGER, PARAMETER :: n = 3, m = 2, h_ne = 4, a_ne = 4
! start problem data
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ), p%X( n ), p%Z( n ), X_stat( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ), p%Y( m ), C_stat( m ) )
   p%new_problem_structure = .TRUE.           ! new structure
   p%n = n ; p%m = m ; p%f = 1.0_wp           ! dimensions & objective constant
   p%G = (/ 1.0_wp, 2.0_wp, 0.0_wp /)         ! objective gradient
   p%C_l = (/ 1.0_wp, 2.0_wp /)               ! constraint lower bound
   p%C_u = (/ 2.0_wp, 2.0_wp /)               ! constraint upper bound
   p%X_l = (/ - 1.0_wp, 3.0_wp, - infinity /) ! variable lower bound
   p%X_u = (/ 1.0_wp, infinity, 2.0_wp /)     ! variable upper bound
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp ! start from zero
!  sparse co-ordinate storage format
   CALL SMT_put( p%A%type, 'COORDINATE', s ) ! Specify co-ordinate storage for A
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! Jacobian A
   p%A%row = (/ 1, 1, 2, 2 /)
   p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
! problem data complete
   CALL LPA_initialize( data, control, inform ) ! Initialize control parameters
   control%infinity = infinity                ! Set infinity
! Solve the problem
   CALL LPA_solve( p, data, control, inform, C_stat = C_stat, X_stat = X_stat )
   IF ( inform%status == 0 ) THEN               !  Successful return
     WRITE( 6, "( ' LPA: ', I0, ' iterations. Optimal objective value =',      &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%iter, inform%obj, p%X
   ELSE                                       !  Error returns
     WRITE( 6, "( ' LPA_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL LPA_terminate( data, control, inform )  !  delete internal workspace
   DEALLOCATE( p%G, p%X_l, p%X_u, p%X, p%Z, X_stat )
   DEALLOCATE( p%C, p%C_l, p%C_u, p%Y, C_stat )
   END PROGRAM GALAHAD_LPA_EXAMPLE
