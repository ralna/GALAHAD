  PROGRAM BARC_test
  USE GALAHAD_BARC_double, only: BARC_projection
  USE GALAHAD_RAND_double
  IMPLICIT NONE    
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, PARAMETER :: n = 10000, n_prob = 1
  INTEGER :: i, prob, max_iter, iter
  REAL ( KIND = wp ) :: b, ts, te
  REAL ( KIND = wp ), DIMENSION( n ) :: C, X_l, X_u, X, A
  TYPE ( RAND_seed ) :: seed
  CALL RAND_initialize( seed )
  X_l = - 1.0_wp
  X_u = 1.0_wp
  max_iter = 0
  CALL CPU_TIME( ts )
  DO prob = 1, n_prob
    DO i = 1, n
      CALL RAND_random_real ( seed, .FALSE., A( i ) )
      CALL RAND_random_real ( seed, .FALSE., C( i ) )
!     C( i ) = 0.1_wp *  C( i )
      C( i ) = 3.0_wp *  C( i )
!     C( i ) = 10000.0_wp *  C( i )
    END DO
    b = 0.001_wp
    CALL BARC_projection( n, C, X_l, X_u, X, iter, A = A, b = b )
    max_iter = max( max_iter, iter )
  END DO
  CALL CPU_TIME( te )
  write(6,"( ' max iter, aver cpu, ', i0, 1X, F5.2 )" ) max_iter,               &
    ( te - ts ) / n_prob
  END PROGRAM BARC_test
