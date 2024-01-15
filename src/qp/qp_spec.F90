! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_QP_EXAMPLE
   USE GALAHAD_KINDS_precision
   USE GALAHAD_QP_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( QP_data_type ) :: data
   TYPE ( QP_control_type ) :: control
   TYPE ( QP_inform_type ) :: inform
   INTEGER ( KIND = ip_ ) :: s
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, m = 2, h_ne = 4, a_ne = 4
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: C_stat, B_stat
! start problem data
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( B_stat( n ), C_stat( m ) )
   p%new_problem_structure = .TRUE.           ! new structure
   p%n = n ; p%m = m ; p%f = 1.0_rp_           ! dimensions & objective constant
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)         ! objective gradient
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)               ! constraint lower bound
   p%C_u = (/ 2.0_rp_, 2.0_rp_ /)               ! constraint upper bound
   p%X_l = (/ - 1.0_rp_, - infinity, - infinity /) ! variable lower bound
   p%X_u = (/ 1.0_rp_, infinity, 2.0_rp_ /)     ! variable upper bound
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_ ! start from zero
!  sparse co-ordinate storage format
   CALL SMT_put( p%H%type, 'COORDINATE', s )     ! Specify co-ordinate
   CALL SMT_put( p%A%type, 'COORDINATE', s )     ! storage for H and A
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%H%val = (/ 1.0_rp_, 2.0_rp_, 1.0_rp_, 3.0_rp_ /) ! Hessian H
   p%H%row = (/ 1, 2, 2, 3 /)                     ! NB lower triangle
   p%H%col = (/ 1, 2, 1, 3 /) ; p%H%ne = h_ne
   p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /) ! Jacobian A
   p%A%row = (/ 1, 1, 2, 2 /)
   p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
! problem data complete
   CALL QP_initialize( data, control, inform )  ! Initialize control parameters
   control%infinity = infinity                  ! Set infinity
   control%quadratic_programming_solver = 'qpa' ! use QPA
   control%scale = 7                            ! Sinkhorn-Knopp scaling
   control%presolve = .TRUE.                    ! Pre-solve the problem
   CALL QP_solve( p, data, control, inform, C_stat, B_stat ) ! Solve
   IF ( inform%status == 0 ) THEN               !  Successful return
     WRITE( 6, "( ' QP: ', I0, ' QPA iterations  ', /,                         &
    &     ' Optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%QPA_inform%iter, inform%obj, p%X
   ELSE                                         !  Error returns
     WRITE( 6, "( ' QP_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL QP_terminate( data, control, inform )  !  delete internal workspace
   END PROGRAM GALAHAD_QP_EXAMPLE
