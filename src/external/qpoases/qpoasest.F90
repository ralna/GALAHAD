! THIS VERSION: GALAHAD 5.6 - 2026-07-22 AT 13:30 GMT.

!  test program for the qpOASES quadratic programming package

!  Nick Gould, July 2026

#include "galahad_modules.h"

PROGRAM qpOASES_example

  USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
  USE GALAHAD_QPOASES_precision, ONLY: qpOASES_solve, qpOASES_options_type

  IMPLICIT NONE

!  local variables

   INTEGER ( KIND = ip_ ), PARAMETER :: out = 6
   REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
   REAL ( KIND = rp_ ), PARAMETER :: infinity = ten ** 20

!  problem parameters

   INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, m = 2, h_ne = 5, a_ne = 4

!  set problem and solution arrays

   REAL ( KIND = rp_ ) :: f, f_sol
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_ptr, A_row
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_ptr, H_row
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val, H_val
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: G, X, X_l, X_u
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y, C_l, C_u
   TYPE( qpOASES_options_type ) :: options
   INTEGER ( KIND = ip_ ) :: status

!  set options and control parameters

   INTEGER( ip_ ) :: iter = 100  ! max number of iterations
   REAL( rp_ ) :: cputime = 3.0_rp_ ! set > 0 to limit CPU time

!  input the problem data per GALAHAD's standard QP format

   ALLOCATE( G( n ), X_l( n ), X_u( n ), X( n ), STAT = status )
   ALLOCATE( C_l( m ), C_u( m ), Y( n + m ), STAT = status )
   ALLOCATE( H_val( h_ne ), H_row( h_ne ), H_ptr( n + 1 ), STAT = status )
   ALLOCATE( A_val( a_ne ), A_row( a_ne ), A_ptr( n + 1 ), STAT = status )

   f = 1.0_rp_                              ! objective constant
   G = [ 0.0_rp_, 2.0_rp_, 0.0_rp_ ]        ! objective gradient
   H_val = [ 1.0_rp_, 2.0_rp_, 1.0_rp_, 1.0_rp_, 3.0_rp_ ] ! Hessian H, column
   H_row = [ 1, 2, 3, 2, 3 ]                   !  storage NB both triangles
   H_ptr = [ 1, 2, 4, 6 ] 
   C_l = [ 1.0_rp_, 2.0_rp_ ]                  ! constraint lower bound
   C_u = [ 2.0_rp_, 2.0_rp_ ]                  ! constraint upper bound
   X_l = [ - 1.0_rp_, - infinity, - infinity ] ! variable lower bound
   X_u = [ 1.0_rp_, infinity, 2.0_rp_ ]        ! variable upper bound
   A_val = [ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ ] ! Jacobian A, column storage
   A_row = [ 1, 1, 2, 2 ]
   A_ptr = [ 1, 2, 4, 5 ]

!  adjust default option values

   options%printLevel = - 1

!  solve the problem

  CALL qpoases_solve( n, m, H_row, H_ptr, H_val, G, A_row, A_ptr, A_val,       &
                      X_l, X_u, C_l, C_u, options, iter, cputime,              &
                      X, Y, f_sol, status )

  IF ( status == 0 ) THEN
    WRITE( out, "( /, ' qpOASES - Fortran interface' )" )
    WRITE( out, "( ' objective function:', ES16.8 )" ) f_sol + f
    WRITE( out, "( ' x:', ( 5ES16.8 ) )" ) X
    WRITE( out, "( ' y:', ( 5ES16.8 ) )" ) Y( n + 1 : )
    WRITE( out, "( ' z:', ( 5ES16.8 ) )" ) Y( : n )
    WRITE( out, "( 1X, I0, ' iterations' ) ") iter
    WRITE( out, "( ' status = ', I0 )" ) status
  ELSE IF ( status == - 199 ) THEN
    WRITE( out, "( ' Error: qpOASES binary not available' )")
  ELSE
    WRITE( out, "( ' Error: Problem not solved to optimality' )")
  END IF

   DEALLOCATE( H_ptr, H_row, H_val, G, X, Y,  STAT = status )
   DEALLOCATE( A_ptr, A_row, A_val, X_l, X_u, C_l, C_u, STAT = status )

  STOP

END PROGRAM qpOASES_example
