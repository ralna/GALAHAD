#include "galahad_modules.h"
  PROGRAM BARC_test !! far from complete
  USE GALAHAD_KINDS_precision
  USE GALAHAD_BARC_precision, ONLY: BARC_projection
  USE GALAHAD_RAND_precision
  IMPLICIT NONE
  INTEGER ( KIND = ip_ ), PARAMETER :: n = 10000, n_prob = 10
  INTEGER ( KIND = ip_ ) :: i, prob, max_iter, iter
  REAL ( KIND = rp_ ) :: b, ts, te
  REAL ( KIND = rp_ ), DIMENSION( n ) :: C, X_l, X_u, X, A
  TYPE ( RAND_seed ) :: seed
  CALL RAND_initialize( seed )
  X_l = - 1.0_rp_
  X_u = 1.0_rp_
  max_iter = 0
  CALL CPU_TIME( ts )
  DO prob = 1, n_prob
    DO i = 1, n
      CALL RAND_random_real ( seed, .FALSE., A( i ) )
      CALL RAND_random_real ( seed, .FALSE., C( i ) )
!     C( i ) = 0.1_rp_ * C( i )
      C( i ) = 3.0_rp_ * C( i )
!     C( i ) = 10000.0_rp_ * C( i )
    END DO
    b = 0.001_rp_
    CALL BARC_projection( n, C, X_l, X_u, X, iter, A = A, b = b )
    max_iter = max( max_iter, iter )
  END DO
  CALL CPU_TIME( te )
  WRITE( 6, "( ' max # iterations, average CPU time = ', i0, ', ', F5.2 )" )   &
     max_iter, ( te - ts ) / n_prob
  WRITE( 6, "( /, ' tests completed' )" )

  END PROGRAM BARC_test
