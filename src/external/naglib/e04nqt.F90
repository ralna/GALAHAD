! THIS VERSION: GALAHAD 5.6 - 2026-10-12 AT 15:00 GMT

#include "galahad_modules.h"

PROGRAM GALAHAD_E04NQF_test

!  main program to test the GALAHAD interface to the NAG convex QP 
!  package E04NQF

!  Nick Gould, October 2026

  USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
  USE GALAHAD_NAGLIB_precision, ONLY: E04NQF_data_type, E04NQF_control_type,   &
                                      E04NQF_inform_type, E04NQF_initialize,   &
                                      E04NQF_solve, E04NQF_terminate,          &
                                      QPT_problem_type, SMT_PUT
  IMPLICIT NONE

!  local variables

   INTEGER ( KIND = ip_ ) :: status
   TYPE ( QPT_problem_type ) :: prob
   TYPE ( E04NQF_data_type ) :: data
   TYPE ( E04NQF_control_type ) :: control
   TYPE ( E04NQF_inform_type ) :: inform

!  problem parameters

   INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, m = 2, h_ne = 4, a_ne = 4
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20

!  results summary output if required (set output_summary > 10) 

   INTEGER ( KIND = ip_ ) :: output_summary = 0
!  INTEGER ( KIND = ip_ ) :: output_summary = 47
!  CHARACTER ( LEN = 10 ) :: summary_filename = 'E04NQ.res'

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
   CALL SMT_put( prob%H%type, 'SPARSE_BY_COLUMNS', status ) ! Specify Hessian
   prob%H%val = [ 1.0_rp_, 2.0_rp_, 1.0_rp_, 3.0_rp_ ]   ! H, column storage
   prob%H%row = [ 1, 2, 3, 3 ]                           ! NB lower triangle
   prob%H%ptr = [ 1, 2, 4, 5 ] 
   CALL SMT_put( prob%A%type, 'SPARSE_BY_COLUMNS', status ) ! Specify Jacobian A
   prob%A%val = [ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ ]      ! column storage
   prob%A%row = [ 1, 1, 2, 2 ]
   prob%A%ptr =  [ 1, 2, 4, 5 ]                             ! NB row pointers
   prob%C_l = [ 1.0_rp_, 2.0_rp_ ]                  ! constraint lower bound
   prob%C_u = [ 2.0_rp_, 2.0_rp_ ]                  ! constraint upper bound
   prob%X_l = [ - 1.0_rp_, - infinity, - infinity ] ! variable lower bound
   prob%X_u = [ 1.0_rp_, infinity, 2.0_rp_ ]        ! variable upper bound

!  initialize the control parameters

   CALL E04NQF_initialize( data, control, inform )

!  solve the problem

   CALL E04NQF_solve( prob, data, control, inform )

!  succesful solve - recover the dual variable as Lagrange multipliers

   WRITE( 6, "( /, ' E04NQF solver' )" )
   IF ( inform%status == 0 ) THEN
     WRITE( 6, "( ' f:', ES16.8 )" ) inform%obj
     WRITE( 6, "( ' x:', ( 5ES16.8 ) )" ) prob%X
     WRITE( 6, "( ' y:', ( 5ES16.8 ) )" ) prob%Y( : m )
     WRITE( 6, "( ' z:', ( 5ES16.8 ) )" ) prob%Z( : n )
     WRITE( 6, "( 1X, I0, ' iterations, status = ', I0 ) ")                    &
       inform%iter, inform%status

!  output a summary of the results to a file if required

    IF ( output_summary > 0 ) THEN
      BACKSPACE( output_summary )
      SELECT CASE ( inform%ifail )
      CASE ( 0, 3, 4 )
        WRITE(  output_summary,                                                &
          "( 1X, I8, 1X, I8, ES16.8, 2ES9.1, bn, I9, I6 )" )                   &
         n, m, inform%obj, inform%primal_infeasibility,                        &
         inform%dual_infeasibility, inform%iter, inform%ifail
      CASE DEFAULT
        WRITE(  output_summary,                                                &
          "( 1X, I8, 1X, I8, ES16.8, 2ES9.1, bn, I9, I6 )" )                   &
          n, m, inform%obj, inform%primal_infeasibility,                       &
          inform%dual_infeasibility, - inform%iter, inform%ifail
      END SELECT
    END IF

!  unsucessful solve

   ELSE
     WRITE( 6, "( ' Error return: status = ', I0 )" ) inform%status
   END IF

!  clean up after the solve

   CALL E04NQF_terminate( data, control, inform )
   DEALLOCATE( prob%X_l, prob%X_u, prob%C_l, prob%C_u, prob%X, prob%Y, prob%Z, &
               prob%G, prob%R, prob%C, STAT = status )
   DEALLOCATE( prob%A%type, prob%A%ptr, prob%A%col, prob%A%val, STAT = status )
   DEALLOCATE( prob%H%type, prob%H%val, prob%H%row, prob%H%col, STAT = status )
   STOP

  END PROGRAM GALAHAD_E04NQF_test

