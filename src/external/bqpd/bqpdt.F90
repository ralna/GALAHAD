! THIS VERSION: GALAHAD 5.6 - 2026-10-06 AT 07:45 GMT
! Nick Gould (nick.gould@stfc.ac.uk)

#include "galahad_modules.h"

PROGRAM GALAHAD_BQPD_example

  USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
  USE GALAHAD_BQPD_precision, ONLY: BQPD_data_type, BQPD_control_type,         &
                                    BQPD_inform_type, BQPD_initialize,         &
                                    BQPD_solve, BQPD_terminate,                &
                                    QPT_problem_type, SMT_PUT
  IMPLICIT NONE

!  local variables

   INTEGER ( KIND = ip_ ) :: status
   TYPE ( QPT_problem_type ) :: prob
   TYPE ( BQPD_data_type ) :: data
   TYPE ( BQPD_control_type ) :: control
   TYPE ( BQPD_inform_type ) :: inform

!  problem parameters

   INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, m = 2, h_ne = 4, a_ne = 4
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20

!  input the problem data per GALAHAD's standard QP format

   ALLOCATE( prob%G( n ), prob%X_l( n ), prob%X_u( n ), STAT = status )
   ALLOCATE( prob%C_l( m ), prob%C_u( m ), prob%X( n ), STAT = status )
   ALLOCATE( prob%H%val( h_ne ), prob%H%row( h_ne ), prob%H%col( h_ne ),       &
             STAT = status )
   ALLOCATE( prob%A%val( a_ne ), prob%A%col( a_ne ), prob%A%ptr( n + 1 ),      &
             STAT = status )

   prob%n = n ; prob%m = m ; prob%A%ne = a_ne ; prob%H%ne = h_ne
   prob%f = 1.0_rp_                              ! objective constant
   prob%G = [ 0.0_rp_, 2.0_rp_, 0.0_rp_ ]        ! objective gradient
   CALL SMT_put( prob%H%type, 'COORDINATE', status )   ! Specify co-ordinate
   prob%H%val = [ 1.0_rp_, 2.0_rp_, 1.0_rp_, 3.0_rp_ ] ! Hessian H, coordinate
   prob%H%row = [ 1, 2, 2, 3 ]                         ! store NB upper triangle
   prob%H%col = [ 1, 2, 3, 3 ]
!  prob%H%ptr = [ 1, 2, 3, 5 ] 
   CALL SMT_put( prob%A%type, 'SPARSE_BY_ROWS', status ) ! storage for H and A
   prob%A%val = [ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ ] ! Jacobian A, row storage
!  prob%A%row = [ 1, 1, 2, 2 ]
   prob%A%col = [ 1, 2, 2, 3 ]
!  prob%A%ptr_col = [ 1, 2, 4, 5 ]
   prob%A%ptr = [ 1, 3, 5 ]                         ! NB row pointers
   prob%C_l = [ 1.0_rp_, 2.0_rp_ ]                  ! constraint lower bound
   prob%C_u = [ 2.0_rp_, 2.0_rp_ ]                  ! constraint upper bound
   prob%X_l = [ - 1.0_rp_, - infinity, - infinity ] ! variable lower bound
   prob%X_u = [ 1.0_rp_, infinity, 2.0_rp_ ]        ! variable upper bound

!  solve the problem

  CALL BQPD_initialize( data, control, inform )
  CALL BQPD_solve( prob, data, control, inform )

!  succesful solve - recover the dual variable as Lagrange multipliers

  WRITE( 6, "( /, ' BQPD solver' )" )
  IF ( inform%status == 0 ) THEN
    WRITE( 6, "( ' f:', ES16.8 )" ) inform%obj
    WRITE( 6, "( ' x:', ( 5ES16.8 ) )" ) prob%X
    WRITE( 6, "( ' y:', ( 5ES16.8 ) )" ) prob%Y( : m )
    WRITE( 6, "( ' z:', ( 5ES16.8 ) )" ) prob%Z( : n )
    WRITE( 6, "( 1X, I0, ' iterations, status = ' I0 ) ")                      &
      inform%iter, inform%status

!  unsucessful solve

  ELSE
    WRITE( 6, "( ' Error return: status = ', I0 )" ) inform%status
  END IF

!  clean up after the solve

  CALL BQPD_terminate( data, control, inform )
  DEALLOCATE( prob%X_l, prob%X_u, prob%C_l, prob%C_u, prob%X, prob%Y, prob%Z,  &
              STAT = status )
  DEALLOCATE( prob%A%type, prob%A%ptr, prob%A%col, prob%A%val, STAT = status )
  DEALLOCATE( prob%H%type, prob%H%val, prob%H%row, prob%H%col, STAT = status )
  STOP

END PROGRAM GALAHAD_BQPD_example

