      SUBROUTINE MC29AD( m, n, ne, A, IRN, ICN, R, C, W, lp, ifail )
      INTEGER :: m, n, ne
      DOUBLE PRECISION :: A( ne )
      INTEGER :: IRN( ne ),ICN( ne )
      DOUBLE PRECISION :: R( m ),C( n ), W( m * 2 + n * 3 )
      INTEGER :: lp, ifail
      END SUBROUTINE MC29AD 
