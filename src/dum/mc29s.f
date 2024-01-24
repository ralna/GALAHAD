! THIS VERSION: GALAHAD 4.3 - 2024-01-24 AT 15:50 GMT.

      SUBROUTINE MC29A( m, n, ne, A, IRN, ICN, R, C, W, lp, ifail )
      USE GALAHAD_KINDS
      INTEGER ( KIND = ip_ ) :: m, n, ne
      REAL :: A( ne )
      INTEGER ( KIND = ip_ ) :: IRN( ne ), ICN( ne )
      REAL :: R( m ), C( n ), W( m * 2 + n * 3 )
      INTEGER ( KIND = ip_ ) :: lp, ifail
      END SUBROUTINE MC29A
