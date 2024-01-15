! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_QPT_test_deck
   USE GALAHAD_KINDS_precision
   USE GALAHAD_QPT_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20 !solver-dependent
   TYPE ( QPT_problem_type ) :: p
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, m = 2, h_ne = 4, a_ne = 4
! start problem data
   ALLOCATE( p%name( 6 ) )
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   p%name = TRANSFER( 'QPprob', p%name )      ! name
   p%new_problem_structure = .TRUE.           ! new structure
   p%Hessian_kind = - 1 ; p%gradient_kind = - 1 ! generic quadratic program
   p%n = n ; p%m = m ; p%f = 1.0_rp_           ! dimensions & objective constant
   p%G = (/ 0.0_rp_,  2.0_rp_,  0.0_rp_ /)         ! objective gradient
   p%C_l = (/ 1.0_rp_,  2.0_rp_ /)               ! constraint lower bound
   p%C_u = (/ 2.0_rp_,  2.0_rp_ /)               ! constraint upper bound
   p%X_l = (/ - 1.0_rp_,  - infinity, - infinity /) ! variable lower bound
   p%X_u = (/ 1.0_rp_,  infinity, 2.0_rp_ /)     ! variable upper bound
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_ ! start from zero
!  sparse co-ordinate storage format
   CALL QPT_put_H( p%H%type, 'COORDINATE' )  ! Specify co-ordinate
   CALL QPT_put_A( p%A%type, 'COORDINATE' )  ! storage for H and A
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%H%val = (/ 1.0_rp_,  2.0_rp_,  3.0_rp_,  4.0_rp_ /) ! Hessian H
   p%H%row = (/ 1, 2, 3, 3 /)                     ! NB lower triangle
   p%H%col = (/ 1, 2, 3, 1 /) ; p%H%ne = h_ne
   p%A%val = (/ 2.0_rp_,  1.0_rp_,  1.0_rp_,  1.0_rp_ /) ! Jacobian A
   p%A%row = (/ 1, 1, 2, 2 /)
   p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
! problem data complete
! now call minimizer ....
! ...
! ... minimization call completed. Deallocate arrays
   DEALLOCATE( p%name, p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z )
   DEALLOCATE( p%H%val, p%H%row, p%H%col, p%H%type )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%type )
   WRITE( 6, "( ' run terminated successfully ' )" )
   END PROGRAM GALAHAD_QPT_test_deck
