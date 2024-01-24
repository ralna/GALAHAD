! THIS VERSION: GALAHAD 4.3 - 2024-01-24 AT 15:55 GMT.

      SUBROUTINE MC29AD( m, n, ne, A, IRN, ICN, R, C, W, lp, ifail )
      USE GALAHAD_KINDS
      INTEGER ( KIND = ip_ ) :: m, n, ne
      DOUBLE PRECISION :: A( ne )
      INTEGER ( KIND = ip_ ) :: IRN( ne ), ICN( ne )
      DOUBLE PRECISION :: R( m ), C( n ), W( m * 2 + n * 3 )
      INTEGER ( KIND = ip_ ) :: lp, ifail
      END SUBROUTINE MC29AD
