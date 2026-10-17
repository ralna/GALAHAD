! THIS VERSION: GALAHAD 5.6 - 2026-10-07 AT 15:00 GMT

#include "galahad_modules.h"

PROGRAM GALAHAD_OSQP_test

!  main program to test the GALAHAD interface to the OSQP convex QP package

!  Nick Gould, October 2026

  USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
  USE GALAHAD_OSQP_precision, ONLY: OSQP_data_type, OSQP_control_type,         &
                                    OSQP_inform_type, OSQP_initialize,         &
                                    OSQP_solve, OSQP_terminate,                &
                                    QPT_problem_type, SMT_PUT
  IMPLICIT NONE

!  local variables

   INTEGER ( KIND = ip_ ) :: status
   TYPE ( QPT_problem_type ) :: prob
   TYPE ( OSQP_data_type ) :: data
   TYPE ( OSQP_control_type ) :: control
   TYPE ( OSQP_inform_type ) :: inform

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
   CALL SMT_put( prob%H%type, 'SPARSE_BY_ROWS', status ) ! Specify Hessian
   prob%H%val = [ 1.0_rp_, 2.0_rp_, 1.0_rp_, 3.0_rp_ ]   ! H, column storage
   prob%H%row = [ 1, 2, 2, 3 ]                           ! NB upper triangle =
   prob%H%ptr = [ 1, 2, 3, 5 ]                           ! lower triangle byrows
   CALL SMT_put( prob%A%type, 'SPARSE_BY_COLUMNS', status ) ! Specify Jacobian A
   prob%A%val = [ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ ]      ! column storage
   prob%A%row = [ 1, 1, 2, 2 ]
   prob%A%ptr =  [ 1, 2, 4, 5 ]                            ! NB row pointers
   prob%C_l = [ 1.0_rp_, 2.0_rp_ ]                  ! constraint lower bound
   prob%C_u = [ 2.0_rp_, 2.0_rp_ ]                  ! constraint upper bound
   prob%X_l = [ - 1.0_rp_, - infinity, - infinity ] ! variable lower bound
   prob%X_u = [ 1.0_rp_, infinity, 2.0_rp_ ]        ! variable upper bound

!  initialize the control parameters

   CALL OSQP_initialize( data, control, inform )

!  assign non-default settings prior to solution

   control%settings%verbose = 0

!  Change linear system solver to a direct one

   control%settings%linsys_solver = 1

!  solve the problem

   CALL OSQP_solve( prob, data, control, inform )

!  succesful solve - recover the dual variable as Lagrange multipliers

   WRITE( 6, "( /, ' OSQP solver' )" )
   IF ( inform%status == 0 ) THEN
     WRITE( 6, "( ' f:', ES16.8 )" ) inform%obj
     WRITE( 6, "( ' x:', ( 5ES16.8 ) )" ) prob%X
     WRITE( 6, "( ' y:', ( 5ES16.8 ) )" ) prob%Y( : m )
     WRITE( 6, "( ' z:', ( 5ES16.8 ) )" ) prob%Z( : n )
     WRITE( 6, "( 1X, I0, ' iterations, status = ', I0 ) ")                    &
       inform%iter, inform%status

!  unsucessful solve

   ELSE
     WRITE( 6, "( ' Error return: status = ', I0 )" ) inform%status
   END IF

!  clean up after the solve

   CALL OSQP_terminate( data, control, inform )
   DEALLOCATE( prob%X_l, prob%X_u, prob%C_l, prob%C_u, prob%X, prob%Y, prob%Z, &
               prob%G, STAT = status )
   DEALLOCATE( prob%A%type, prob%A%ptr, prob%A%col, prob%A%val, STAT = status )
   DEALLOCATE( prob%H%type, prob%H%val, prob%H%row, prob%H%col, STAT = status )
   STOP

END PROGRAM GALAHAD_OSQP_test
 
