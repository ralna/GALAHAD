      SUBROUTINE MC29A( m, n, ne, A, IRN, ICN, R, C, W, lp, ifail )
      INTEGER :: m, n, ne
      REAL :: A( ne )
      INTEGER :: IRN( ne ),ICN( ne )
      REAL :: R( m ),C( n ), W( m * 2 + n * 3 )
      INTEGER :: lp, ifail
      END SUBROUTINE MC29A
