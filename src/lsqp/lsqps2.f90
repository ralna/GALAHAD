! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
   PROGRAM GALAHAD_LSQP_EXAMPLE
   USE GALAHAD_LSQP_double                       ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( LSQP_data_type ) :: data
   TYPE ( LSQP_control_type ) :: control        
   TYPE ( LSQP_inform_type ) :: info
   INTEGER, PARAMETER :: n = 16, m = 8, a_ne = 53
   INTEGER :: i, s




! start problem data
   ALLOCATE( p%X_l( n ), p%X_u( n ), p%G( n ), p%WEIGHT( n )  )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%new_problem_structure = .TRUE.           ! new structure
!  p%gradient_kind = 2
!  p%Hessian_kind = 2
   p%gradient_kind = 2
   p%Hessian_kind = 2
   p%n = n ; p%m = m ; p%f = 0.0_wp           ! dimensions & objective constant
   CALL SMT_put( p%A%type, 'COORDINATE', s )  ! storage for H and A
   p%A%ne = a_ne
   READ( 5, "( ( 5ES12.4 ) )" ) p%G( : p%n )
!  p%G = 0.01 * p%G
   READ( 5, "( ( 5ES12.4 ) )" ) p%WEIGHT( : p%n )
   p%WEIGHT = SQRT( p%WEIGHT ) ;
   READ( 5, "( ( 5ES12.4 ) )" ) p%X_l( : p%n )
   READ( 5, "( ( 5ES12.4 ) )" ) p%X_u( : p%n )
   READ( 5, "( ( 2( 2I8, ES12.4 ) ) )" )                      &
     ( p%A%row( i ), p%A%col( i ), p%A%val( i ), i = 1, p%A%ne )
   READ( 5, "( 5ES12.4 ) )" ) p%C_l( : p%m )
   READ( 5, "( 5ES12.4 ) )" ) p%C_u( : p%m )
   
! integer components complete   
   CALL LSQP_initialize( data, control )      ! Initialize control parameters
   control%infinity = infinity                ! Set infinity
   control%restore_problem = 1                ! Restore vector data on exit
   control%print_level = 1
   control%maxit = 1000
!  control%muzero = 0.0001
   p%X = 0.0_wp
   p%Y = 0.0_wp ; p%Z = 1000.0_wp                   ! start multipliers from zero
! sparse co-ordinate storage format: real components
! real components complete   
     ALLOCATE( p%X0( n ) ) ; p%X0 = 0.0
     CALL LSQP_solve( p, data, control, info )     ! Solve problem
     IF ( info%status == 0 ) THEN                  ! Successful return
         WRITE( 6, "( ' Eg ', I6, ' iterations. Optimal objective value =',&
        &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )            &
         info%iter, info%obj, p%X
     ELSE                                          !  Error returns
       WRITE( 6, "( ' LSQP_solve exit status = ', I6 ) " ) info%status
     END IF
   CALL LSQP_terminate( data, control, info )      !  delete internal workspace
   END PROGRAM GALAHAD_LSQP_EXAMPLE

