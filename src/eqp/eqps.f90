! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
   PROGRAM GALAHAD_EQP_EXAMPLE
   USE GALAHAD_EQP_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( QPT_problem_type ) :: p
   TYPE ( EQP_data_type ) :: data
   TYPE ( EQP_control_type ) :: control        
   TYPE ( EQP_inform_type ) :: inform
   INTEGER :: s
   INTEGER, PARAMETER :: n = 3, m = 2, h_ne = 4, a_ne = 4 
! start problem data
   ALLOCATE( p%G( n ), p%C( m ), p%X( n ), p%Y( m ) )
   p%new_problem_structure = .TRUE.           ! new structure
   p%n = n ; p%m = m ; p%f = 1.0_wp           ! dimensions & objective constant
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)         ! objective gradient
   p%C = (/ - 2.0_wp, - 2.0_wp /)             ! constraint constants
   p%X = 0.0_wp ; p%Y = 0.0_wp                ! start from zero
! sparse co-ordinate storage format
   CALL SMT_put( p%H%type, 'COORDINATE', s )  ! Specify co-ordinate 
   CALL SMT_put( p%A%type, 'COORDINATE', s )  ! storage for H and A
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%H%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /) ! Hessian H
   p%H%row = (/ 1, 2, 3, 3 /)                     ! NB lower triangle
   p%H%col = (/ 1, 2, 3, 1 /) ; p%H%ne = h_ne
   p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! Jacobian A
   p%A%row = (/ 1, 1, 2, 2 /)
   p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
! problem data complete   
   CALL EQP_initialize( data, control, inform ) ! Initialize control parameters
   CALL EQP_solve( p, data, control, inform )   !  Solve problem
   IF ( inform%status == 0 ) THEN               !  Successful return
     WRITE( 6, "( ' EQP: ', I0, ' CG iteration(s). Optimal objective value =', &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%cg_iter, inform%obj, p%X
   ELSE                                         !  Error returns
     WRITE( 6, "( ' EQP_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL EQP_terminate( data, control, inform )  !  delete internal workspace
   DEALLOCATE( p%G, p%C, p%X, p%Y )             !  deallocate problem arrays
   DEALLOCATE( p%H%val, p%H%row, p%H%col, p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%H%type, p%A%type )
   END PROGRAM GALAHAD_EQP_EXAMPLE
