! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_SEC_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SEC_precision
   USE GALAHAD_RAND_precision
   IMPLICIT NONE
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 5
   TYPE ( SEC_control_type ) :: control
   TYPE ( SEC_inform_type ) :: inform
   REAL ( KIND = rp_ ), DIMENSION( n ) :: S, Y, W
   REAL ( KIND = rp_ ), DIMENSION( n * ( n + 1 ) / 2 ) :: H
   INTEGER ( KIND = ip_ ) :: iter, fail
   TYPE ( RAND_seed ) :: seed

   WRITE( 6, "( ' Error return tests -' )")
   CALL SEC_initialize( control, inform ) !  initialize data
   CALL SEC_initial_approximation( n, H, control, inform )
   S = 1.0_rp_
   Y = - 1.0_rp_
   CALL SEC_bfgs_update( n, S, Y, H, W, control, inform ) ! apply BFGS update
   IF ( inform%status /= 0 ) WRITE( 6, "( ' BFGS error status = ', I0 )" )     &
     inform%status
   Y = S
   CALL SEC_sr1_update( n, S, Y, H, W, control, inform ) ! apply BFGS update
   IF ( inform%status /= 0 ) WRITE( 6, "( ' SR1 error status = ', I0 )" )     &
     inform%status

   WRITE( 6, "( /, ' Normal tests - ' )")
   CALL RAND_initialize( seed ) ! Initialize the random generator word
   CALL SEC_initialize( control, inform ) !  initialize data
   CALL SEC_initial_approximation( n, H, control, inform )
   fail = 0 ! count the failures
   DO iter = 1, 5 * n
     CALL RAND_random_real( seed, .FALSE., S )  ! pick random S and Y
     CALL RAND_random_real( seed, .FALSE., Y )
     IF ( DOT_PRODUCT( S, Y ) < 0.0_rp_ ) Y = - Y ! ensure that S'Y is positive
     CALL SEC_bfgs_update( n, S, Y, H, W, control, inform ) ! apply BFGS update
     IF ( inform%status /= 0 ) fail = fail + 1
   END DO
   IF ( fail == 0 ) THEN  ! check for overall success
     WRITE( 6, "( ' SEC - BFGS: no failures ' )" )
   ELSE
     WRITE( 6, "( ' SEC - BFGS: ', I0, ' failures ' )" ) fail
   END IF
!  CALL SEC_initial_approximation( n, H, control, inform )
   fail = 0 ! count the failures
   DO iter = 1, 5 * n
     CALL RAND_random_real( seed, .FALSE., S )  ! pick random S and Y
     CALL RAND_random_real( seed, .FALSE., Y )
     IF ( DOT_PRODUCT( S, Y ) < 0.0_rp_ ) Y = - Y ! ensure that S'Y is positive
     CALL SEC_sr1_update( n, S, Y, H, W, control, inform ) ! apply SR1 update
     IF ( inform%status /= 0 ) fail = fail + 1
   END DO
   IF ( fail == 0 ) THEN  ! check for overall success
     WRITE( 6, "( ' SEC - SR1: no failures ' )" )
   ELSE
     WRITE( 6, "( ' SEC - SR1: ', I0, ' failures ' )" ) fail
   END IF
   END PROGRAM GALAHAD_SEC_test
