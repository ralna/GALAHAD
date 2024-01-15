! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_QPC_EXAMPLE
   USE GALAHAD_KINDS_precision
   USE GALAHAD_QPC_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( QPC_data_type ) :: data
   TYPE ( QPC_control_type ) :: control
   TYPE ( QPC_inform_type ) :: info
   INTEGER ( KIND = ip_ ) :: s
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, m = 2, h_ne = 4, a_ne = 4
   INTEGER ( KIND = ip_ ) :: data_storage_type = 0
   INTEGER ( KIND = ip_ ), DIMENSION( m ) :: C_stat
   INTEGER ( KIND = ip_ ), DIMENSION( n ) :: B_stat
! start problem data
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   p%new_problem_structure = .TRUE.           ! new structure
   p%n = n ; p%m = m ; p%f = 1.0_rp_           ! dimensions & objective constant
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)         ! objective gradient
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)               ! constraint lower bound
   p%C_u = (/ 2.0_rp_, 2.0_rp_ /)               ! constraint upper bound
   p%X_l = (/ - 1.0_rp_, - infinity, - infinity /) ! variable lower bound
   p%X_u = (/ 1.0_rp_, infinity, 2.0_rp_ /)     ! variable upper bound
   p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_ ! start from zero
! sparse co-ordinate storage format
   IF ( data_storage_type == 0 ) THEN
   CALL SMT_put( p%H%type, 'COORDINATE', s )  ! Specify co-ordinate
   CALL SMT_put( p%A%type, 'COORDINATE', s )  ! storage for H and A
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /) ! Hessian H
   p%H%row = (/ 1, 2, 3, 3 /)                     ! NB lower triangle
   p%H%col = (/ 1, 2, 3, 1 /) ; p%H%ne = h_ne
   p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /) ! Jacobian A
   p%A%row = (/ 1, 1, 2, 2 /)
   p%A%col = (/ 1, 2, 2, 3 /) ; p%A%ne = a_ne
! sparse row-wise storage format
   ELSE IF ( data_storage_type == - 1 ) THEN
   CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', s )  ! Specify sparse-by-rows
   CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', s )  ! storage for H and A
   ALLOCATE( p%H%val( h_ne ), p%H%col( h_ne ), p%H%ptr( n + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%col( a_ne ), p%A%ptr( m + 1 ) )
   p%H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /) ! Hessian H
   p%H%col = (/ 1, 2, 3, 1 /)                     ! NB lower triangular
   p%H%ptr = (/ 1, 2, 3, 5 /)                     ! Set row pointers
   p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /) ! Jacobian A
   p%A%col = (/ 1, 2, 2, 3 /)
   p%A%ptr = (/ 1, 3, 5 /)                        ! Set row pointers
! dense storage format
   ELSE
   CALL SMT_put( p%H%type, 'DENSE', s )  ! Specify dense
   CALL SMT_put( p%A%type, 'DENSE', s )  ! storage for H and A
   ALLOCATE( p%H%val( n * ( n + 1 ) / 2 ) )
   ALLOCATE( p%A%val( n * m ) )
   p%H%val = (/ 1.0_rp_, 0.0_rp_, 2.0_rp_, 4.0_rp_, 0.0_rp_, 3.0_rp_ /)!Hessian
   p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)!Jacobian
! problem data complete
   END IF
   CALL QPC_initialize( data, control, info )    ! Initialize control parameters
   control%infinity = infinity                   ! Set infinity
!  control%print_level = 1
   CALL QPC_solve( p,  C_stat, B_stat, data, control, info )    ! Solve problem
   IF ( info%status == 0 ) THEN                  !  Successful return
     WRITE( 6, "( ' QPC: ', I0, ' QPA and ', I0, ' QPB iterations  ', /,       &
    &     ' Optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     info%QPA_inform%iter, info%QPB_inform%iter, info%obj, p%X
   ELSE                                          !  Error returns
     WRITE( 6, "( ' QPC_solve exit status = ', I6 ) " ) info%status
   END IF
   CALL QPC_terminate( data, control, info )    !  delete internal workspace
   END PROGRAM GALAHAD_QPC_EXAMPLE

