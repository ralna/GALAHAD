! THIS VERSION: 25/04/2022 AT 13:45 GMT
! Nick Gould (nick.gould@stfc.ac.uk)

#include "galahad_modules.h"

PROGRAM BQPD_example

  USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
  IMPLICIT NONE

!  local variables

   INTEGER ( KIND = ip_ ) :: i, j, k, peq, kmax, mode
   INTEGER ( KIND = ip_ ), PARAMETER :: out = 6
   REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
   REAL ( KIND = rp_ ), PARAMETER :: infinity = ten ** 20
   REAL ( KIND = rp_ ) :: f_min = - infinity

!  problem parameters

   INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, m = 2, h_ne = 4, a_ne = 4
   INTEGER ( KIND = ip_ ), PARAMETER :: np1 = n + 1, npm = n + m
   INTEGER ( KIND = ip_ ), PARAMETER :: maxa = a_ne + n
   INTEGER ( KIND = ip_ ), PARAMETER :: mlp = 100
   INTEGER ( KIND = ip_ ), PARAMETER :: nprof = 100000

!  set problem and solution arrays

   REAL ( KIND = rp_ ) :: f, f_opt
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_ptr, A_col, IGA
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_row, H_col
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: LP, LS, LWS
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: G, X, X_l, X_u
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: C_l, C_u, ALP
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: B_l, B_u, R, E, W, WS
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val, H_val, GA
   INTEGER ( KIND = ip_ ), DIMENSION( 1 ) :: INFO
!  TYPE ( BQPD_settings_type ) :: settings
!  TYPE ( BQPD_data_type ) :: data
   INTEGER ( KIND = ip_ ) :: status

!  common blocks (ouch!)

   INTEGER :: len_ws, len_lws, len_ws_gdotx, len_lws_gdotx, len_ws_sparsel
   INTEGER :: len_lws_sparsel, len_ws_bqpd, len_lws_bqpd, print_level, iprint
   COMMON / noutc / iprint
   COMMON / iprintc / print_level
   COMMON / wsc / len_ws_gdotx, len_lws_gdotx, len_ws_bqpd, len_lws_bqpd,      &
                  len_ws, len_lws

!  input the problem data per GALAHAD's standard QP format

   ALLOCATE( G( n ), X_l( n ), X_u( n ), STAT = status )
   ALLOCATE( C_l( m ), C_u( m ), STAT = status )
   ALLOCATE( H_val( h_ne ), H_row( h_ne ), H_col( h_ne ), STAT = status )
   ALLOCATE( A_val( a_ne ), A_col( a_ne ), A_ptr( np1 ), STAT = status )

   f = 1.0_rp_                              ! objective constant
   G = [ 0.0_rp_, 2.0_rp_, 0.0_rp_ ]        ! objective gradient
   H_val = [ 1.0_rp_, 2.0_rp_, 1.0_rp_, 3.0_rp_ ] ! Hessian H, coordinate store
   H_row = [ 1, 2, 2, 3 ]                         ! NB upper triangle
   H_col = [ 1, 2, 3, 3 ]
!  H_ptr = [ 1, 2, 3, 5 ] 
   A_val = [ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ ] ! Jacobian A, row storage
!  A_row = [ 1, 1, 2, 2 ]
   A_col = [ 1, 2, 2, 3 ]
!  A_ptr_col = [ 1, 2, 4, 5 ]
   A_ptr = [ 1, 3, 5 ]                         ! NB row pointers
   C_l = [ 1.0_rp_, 2.0_rp_ ]                  ! constraint lower bound
   C_u = [ 2.0_rp_, 2.0_rp_ ]                  ! constraint upper bound
   X_l = [ - 1.0_rp_, - infinity, - infinity ] ! variable lower bound
   X_u = [ 1.0_rp_, infinity, 2.0_rp_ ]        ! variable upper bound

!  transfer the data into BQPD's QP format

   ALLOCATE( GA( maxa ), IGA( 0 : maxa + m + 3 ), STAT = status )
   ALLOCATE( ALP( mlp ), LP( mlp ), STAT = status )
   ALLOCATE( B_l( npm ), B_u( npm ), X( n ), STAT = status )

!  reset problem constraint data (NB 1-based integer index arrays)

   B_l( : n ) = X_l( : n ) ; B_u( : n ) = X_u( : n )
   B_l( np1 : npm ) = C_l( 1 : m ) ; B_u( np1 : npm ) = C_u( 1 : m )
   IGA( 0 ) = maxa + 1

!  include the gradient g in GA and IGA

   GA( 1 : n ) = G( 1 : n )
   IGA( 1 : n ) = [ ( i, i = 1, n ) ]

!  include the Jacobian A in GA and IGA

   GA( n + 1 : n + a_ne ) = A_val
   IGA( n + 1 : n + a_ne ) = A_col
   IGA( n + a_ne + 1 ) = 1
   IGA( n + a_ne + 2 : n + a_ne + m + 2 ) = A_ptr + n

!  remove unneeded problem data

   DEALLOCATE( X_l, X_u, C_l, C_u, A_ptr, A_col, A_val, STAT = status )

!  assign non-default settings prior to solution

   mode = 0 ! cold start
   k = 0 ! dimension of reduced space (not used for cold start)
   kmax = MIN( 2000, n ) ! max allowed value of k (not used)
   iprint = print_level

!  allocate and fill bqpd workspace arrays

   len_ws_gdotx = h_ne
   len_lws_gdotx = 2 * h_ne + 1
   len_ws_bqpd = kmax * ( kmax + 9 ) / 2 + npm + m
   len_lws_bqpd = kmax
   len_ws_sparsel = 5 * n + nprof
   len_lws_sparsel = 9 * n + m
   len_ws = len_ws_gdotx + len_ws_bqpd + len_ws_sparsel
   len_lws = len_lws_gdotx + len_lws_bqpd + len_lws_sparsel

   ALLOCATE( R( npm ), W( npm ), E( npm ), LS( npm ), STAT = status )
   ALLOCATE( LWS( len_lws ), WS( len_ws ), STAT = status )
   LWS( 1 ) = h_ne
   LWS( 2 : h_ne + 1 ) = H_row
   LWS( h_ne + 2 : 2 * h_ne + 1 ) = H_col
   WS( 1 : h_ne ) = H_val
   DEALLOCATE( H_val, H_row, H_col, STAT = status )

!  solve the problem

  CALL BQPD( n, m, k, kmax, GA, IGA, X, B_l, B_u, f_opt, f_min,                &
             G, R, W, E, LS, ALP, LP, mlp, peq, WS, LWS, mode,                 &
             status, INFO, print_level, out )

!  succesful solve - recover the dual variable as Lagrange multipliers

  WRITE( out, "( /, ' BQPD solver' )" )
  IF ( status == 0 ) THEN
    DO i = 1, n - k ! active constraints
      j = LS( i )
      IF ( j < 0 ) THEN ! active at upper bound
        R( - j ) = - R( - j )
      END IF
    END DO
    R( ABS( LS( n - k + 1 : npm ) ) ) = 0.0_rp_
    WRITE( out, "( ' objective function:', ES16.8 )" ) f_opt + f
!   WRITE( out, "( ' primal & dual residuals:', 2ES16.8 )" )                   &
!     info%prim_res, info%dual_res
    WRITE( out, "( ' x:', ( 5ES16.8 ) )" ) X
    WRITE( out, "( ' y:', ( 5ES16.8 ) )" ) R( np1 : npm )
    WRITE( out, "( ' z:', ( 5ES16.8 ) )" ) R( : n )
    WRITE( out, "( ' status = ', I0 )" ) status
    WRITE( out, "( 1X, I0, ' iterations' ) ") INFO( 1 )
!   WRITE( out, "( ' active:  ', ( 5I8 ) )" ) LS( : n - k )
!   WRITE( out, "( ' inactive:', ( 5I8 ) )" ) LS( n - k + 1 : npm )

!  unsucessful solve

  ELSE
    WRITE( out, "( ' Error return: status = ', I0 )" ) status
  END IF

!  clean up after the solve

  DEALLOCATE( GA, IGA, X, B_l, B_u, G, R, W, E, LS, ALP, LP, WS, LWS )

  STOP

END PROGRAM BQPD_example

