! THIS VERSION: GALAHAD 2.2 - 23/04/2008 AT 16:30 GMT.
   PROGRAM GALAHAD_LSQP_EXAMPLE
   USE GALAHAD_LSQP_double                       ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( LSQP_data_type ) :: data
   TYPE ( LSQP_control_type ) :: control        
   TYPE ( LSQP_inform_type ) :: inform
   INTEGER, PARAMETER :: n = 3, m = 2, a_ne = 4 
   INTEGER :: i, s
! start problem data
   ALLOCATE( p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   p%new_problem_structure = .TRUE.           ! new structure
   p%n = n ; p%m = m ; p%f = 0.0_wp           ! dimensions & objective constant
   p%C_l = (/ 1.0_wp, 2.0_wp /)               ! constraint lower bound
   p%C_u = (/ 2.0_wp, 2.0_wp /)               ! constraint upper bound
   p%X_l = (/ - 1.0_wp, - infinity, - infinity /) ! variable lower bound
   p%X_u = (/ 1.0_wp, infinity, 2.0_wp /)     ! variable upper bound
   p%gradient_kind = 0   
! sparse co-ordinate storage format: integer components
   CALL SMT_put( p%A%type, 'COORDINATE', s )  ! storage for H and A
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%A%row = (/ 1, 1, 2, 2 /)                 ! Jacobian A
   p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
! integer components complete   
   CALL LSQP_initialize( data, control, inform ) ! Initialize control parameters
   control%infinity = infinity                   ! Set infinity
   control%restore_problem = 1                   ! Restore vector data on exit
!  control%print_level = 1
!  control%SBLS_control%symmetric_linear_solver = 'ma57'
!  control%SBLS_control%print_level = 1
!  control%FDC_control%print_level = 1
   DO i = 0, 2
!  DO i = 0, 1
     p%X = (/  -2.0_wp, 1.0_wp,  3.0_wp /)         ! set x0
     p%Y = 0.0_wp ; p%Z = 0.0_wp                   ! start multipliers from zero
! sparse co-ordinate storage format: real components
     p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! Jacobian A
! real components complete   
     p%Hessian_kind = 2 - i
     IF ( p%Hessian_kind == 0 ) THEN
       control%stop_c = 10.0_wp ** ( - 12 ) ; control%itref_max = 2
     END IF
     IF ( p%Hessian_kind == 2 ) THEN
       ALLOCATE( p%WEIGHT( n ) ) ; p%WEIGHT = (/ 0.1_wp, 1.0_wp, 2.0_wp /)
       ALLOCATE( p%X0( n ) )
     END IF
     IF ( p%Hessian_kind /= 0 ) p%X0 = p%X
     CALL LSQP_solve( p, data, control, inform )   ! Solve problem
     IF ( inform%status == 0 ) THEN                ! Successful return
       IF ( p%Hessian_kind == 0 ) THEN
         WRITE( 6, "( ' Eg ', I1, I6, ' iterations. Optimal potential value =',&
        &       ES12.4, /, ' Analytic center  = ', ( 5ES12.4 ) )" )            &
         i + 1, inform%iter, inform%potential, p%X
       ELSE
         WRITE( 6, "( ' Eg ', I1, I6, ' iterations. Optimal objective value =',&
        &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )            &
         i + 1, inform%iter, inform%obj, p%X
       END IF
     ELSE                                          !  Error returns
       WRITE( 6, "( ' LSQP_solve exit status = ', I6 ) " ) inform%status
     END IF
   END DO
   CALL LSQP_terminate( data, control, inform )    !  delete internal workspace
   END PROGRAM GALAHAD_LSQP_EXAMPLE

