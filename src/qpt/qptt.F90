! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
   PROGRAM GALAHAD_QPT_test_deck
   USE GALAHAD_QPT_double                       ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20 ! solver-dependent
   TYPE ( QPT_problem_type ) :: p
   INTEGER, PARAMETER :: n = 3, m = 2, h_ne = 4, a_ne = 4 
! start problem data
   ALLOCATE( p%name( 6 ) )
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   p%name = TRANSFER( 'QPprob', p%name )      ! name
   p%new_problem_structure = .TRUE.           ! new structure
   p%Hessian_kind = - 1 ; p%gradient_kind = - 1 ! generic quadratic program
   p%n = n ; p%m = m ; p%f = 1.0_wp           ! dimensions & objective constant
   p%G = (/ 0.0_wp, 2.0_wp, 0.0_wp /)         ! objective gradient
   p%C_l = (/ 1.0_wp, 2.0_wp /)               ! constraint lower bound
   p%C_u = (/ 2.0_wp, 2.0_wp /)               ! constraint upper bound
   p%X_l = (/ - 1.0_wp, - infinity, - infinity /) ! variable lower bound
   p%X_u = (/ 1.0_wp, infinity, 2.0_wp /)     ! variable upper bound
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp ! start from zero
!  sparse co-ordinate storage format
   CALL QPT_put_H( p%H%type, 'COORDINATE' )  ! Specify co-ordinate 
   CALL QPT_put_A( p%A%type, 'COORDINATE' )  ! storage for H and A
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%H%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /) ! Hessian H
   p%H%row = (/ 1, 2, 3, 3 /)                     ! NB lower triangle
   p%H%col = (/ 1, 2, 3, 1 /) ; p%H%ne = h_ne
   p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! Jacobian A
   p%A%row = (/ 1, 1, 2, 2 /)
   p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
! problem data complete
! now call minimizer ....
! ...
! ... minimization call completed. Deallocate arrays
   DEALLOCATE( p%name, p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z )
   DEALLOCATE( p%H%val, p%H%row, p%H%col, p%A%val, p%A%row, p%A%col )
   WRITE( 6, "( ' run terminated successfully ' )" )
   END PROGRAM GALAHAD_QPT_test_deck
