! THIS VERSION: GALAHAD 5.6 - 2026-11-27 AT 09:07 GMT.

#include "galahad_modules.h"

!  test the reformulation of a linear program as an equivalent l_1 penalty
!  problem by LPQP, and its restoration

   PROGRAM GALAHAD_LPQP_TEST
   USE GALAHAD_KINDS_precision
   USE GALAHAD_QPT_precision
   USE GALAHAD_LPQP_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: rho = 10.0_rp_
   TYPE ( QPT_problem_type ) :: p
   TYPE ( LPQP_data_type ) :: data
   TYPE ( LPQP_control_type ) :: control
   TYPE ( LPQP_inform_type ) :: inform
   INTEGER ( KIND = ip_ ) :: n, m, smt_stat, test
   REAL ( KIND = rp_ ), DIMENSION( 2 ) :: C_l, C_u

!  minimize 2 x_1 + x_2 + 2 x_3 subject to c_l <= ( x_1 + x_2, x_2 + x_3 )
!  <= c_u and 0 <= x <= 10. In the first test the constraints are bounded
!  on both sides, and the restored problem must have the original bounds.
!  In the second, the first constraint is inconsistent, which must be
!  reported

   n = 3 ; m = 2
   DO test = 1, 2
     IF ( test == 1 ) THEN
       C_l = (/ 1.0_rp_, 1.0_rp_ /) ; C_u = (/ 2.0_rp_, 3.0_rp_ /)
     ELSE
       C_l = (/ 2.0_rp_, 1.0_rp_ /) ; C_u = (/ 1.0_rp_, 3.0_rp_ /)
     END IF
     ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ), p%C( m ), p%C_l( m ),        &
               p%C_u( m ), p%X( n ), p%Y( m ), p%Z( n ) )
     p%n = n ; p%m = m ; p%f = 0.0_rp_
     p%gradient_kind = GALAHAD_GENERAL
     p%G = (/ 2.0_rp_, 1.0_rp_, 2.0_rp_ /)
     p%X_l = 0.0_rp_ ; p%X_u = 10.0_rp_
     p%C_l = C_l ; p%C_u = C_u
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_ ; p%C = 0.0_rp_
     CALL SMT_put( p%A%type, 'COORDINATE', smt_stat ) ; p%A%ne = 4
     ALLOCATE( p%A%row( 4 ), p%A%col( 4 ), p%A%val( 4 ) )
     p%A%row = (/ 1, 1, 2, 2 /) ; p%A%col = (/ 1, 2, 2, 3 /)
     p%A%val = 1.0_rp_
     CALL SMT_put( p%H%type, 'COORDINATE', smt_stat ) ; p%H%ne = 0
     ALLOCATE( p%H%row( 0 ), p%H%col( 0 ), p%H%val( 0 ) )

     CALL LPQP_initialize( control )
     CALL LPQP_formulate( p, rho, .TRUE., data, control, inform )
     IF ( test == 1 ) THEN
       IF ( inform%status /= GALAHAD_ok ) THEN
         WRITE( 6, "( ' range constraint test: LPQP_formulate status = ',     &
        &             I0, ' FAILED' )" ) inform%status
         STOP 1
       END IF
       CALL LPQP_restore( p, data )
       IF ( ANY( p%C_l( : m ) /= C_l ) .OR. ANY( p%C_u( : m ) /= C_u ) ) THEN
         WRITE( 6, "( ' range constraint test: bounds not restored,',         &
        &             ' c_l = ', 2ES9.1, ' c_u = ', 2ES9.1, ' FAILED' )" )     &
           p%C_l( : m ), p%C_u( : m )
         STOP 1
       END IF
       WRITE( 6, "( ' range constraint test: bounds restored' )" )
     ELSE
       IF ( inform%status /= GALAHAD_error_primal_infeasible ) THEN
         WRITE( 6, "( ' inconsistent constraint test: LPQP_formulate',        &
        &             ' status = ', I0, ' FAILED' )" ) inform%status
         STOP 1
       END IF
       WRITE( 6, "( ' inconsistent constraint test: infeasibility',           &
      &           ' reported' )" )
     END IF
     CALL LPQP_terminate( data, control, inform )
     DEALLOCATE( p%G, p%X_l, p%X_u, p%C, p%C_l, p%C_u, p%X, p%Y, p%Z )
     DEALLOCATE( p%A%row, p%A%col, p%A%val, p%A%type )
     DEALLOCATE( p%H%row, p%H%col, p%H%val, p%H%type )
     IF ( ALLOCATED( p%A%ptr ) ) DEALLOCATE( p%A%ptr )
     IF ( ALLOCATED( p%H%ptr ) ) DEALLOCATE( p%H%ptr )
   END DO
   WRITE( 6, "( /, ' tests completed' )" )
   END PROGRAM GALAHAD_LPQP_TEST
