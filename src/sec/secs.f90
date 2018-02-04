! THIS VERSION: GALAHAD 3.0 - 17/08/2017 AT 11:40 GMT.
   PROGRAM GALAHAD_SEC_example
   USE GALAHAD_SEC_double                    ! double precision version
   USE GALAHAD_rand_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, PARAMETER :: n = 5
   TYPE ( SEC_control_type ) :: control
   TYPE ( SEC_inform_type ) :: inform
   REAL ( KIND = wp ), DIMENSION( n ) :: S, Y, W
   REAL ( KIND = wp ), DIMENSION( n * ( n + 1 ) / 2 ) :: H
   INTEGER :: iter, fail
   REAL ( KIND = wp ) :: delta, lambda
   TYPE ( RAND_seed ) :: seed
   CALL RAND_initialize( seed ) ! Initialize the random generator word
   CALL SEC_initialize( control, inform ) !  initialize data
   CALL SEC_initial_approximation( n, H, control, inform )
   fail = 0 ! count the failures
   DO iter = 1, 5 * n
     CALL RAND_random_real( seed, .FALSE., S )  ! pick random S and Y
     CALL RAND_random_real( seed, .FALSE., Y )
     IF ( DOT_PRODUCT( S, Y ) < 0.0_wp ) Y = - Y ! ensure that S^T Y is positive
     CALL SEC_bfgs_update( n, S, Y, H, W, control, inform ) ! apply BFGS update
     IF ( inform%status /= 0 ) fail = fail + 1
   END DO
   IF ( fail == 0 ) THEN  ! check for overall success
     WRITE( 6, "( ' SEC: no failures ' )" )
   ELSE
     WRITE( 6, "( ' SEC: ', I0, ' failures ' )" ) fail
   END IF
   END PROGRAM GALAHAD_SEC_example
