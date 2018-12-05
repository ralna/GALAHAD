      SUBROUTINE LA15ID( ICNTL, CNTL, KEEP )
      DOUBLE PRECISION :: CNTL( 3 )
      INTEGER :: ICNTL( 3 )
      INTEGER :: KEEP( 7 )
      END SUBROUTINE LA15ID

      SUBROUTINE LA15AD( A, IND, nzero, ia, n, IP, IW, W, g, u,
     &                   ICNTL, CNTL, KEEP )
      DOUBLE PRECISION :: g, U
      INTEGER :: ia, n, nzero
      DOUBLE PRECISION :: A( ia ),W( n )
      INTEGER :: IND( ia, 2 ),IP(N,2),IW( n, 8 )
      DOUBLE PRECISION :: CNTL( 3 )
      INTEGER :: ICNTL( 3 )
      INTEGER :: KEEP( 7 )
      END SUBROUTINE LA15AD

      SUBROUTINE LA15BD( A, IND, ia, n, IP, IW, W, g, B,
     &                   trans, ICNTL, KEEP )
      DOUBLE PRECISION :: g
      INTEGER :: ia, n
      LOGICAL :: trans
      DOUBLE PRECISION :: A( ia ), B( n ), W( n )
      INTEGER :: IND( ia, 2 ),IP( n, 2 ), IW( n, 4 )
      INTEGER :: ICNTL( 3 )
      INTEGER :: KEEP( 7 )
      END SUBROUTINE LA15BD

      SUBROUTINE LA15CD( A, IND, ia, n, IP, IW, W, g, u, mm,
     &                   ICNTL, CNTL, KEEP )
      DOUBLE PRECISION :: g, u
      INTEGER :: ia, mm, n
      DOUBLE PRECISION :: A( ia ),W( n )
      INTEGER IND( ia, 2 ), IP( n, 2 ), IW( n, 4 )
      DOUBLE PRECISION :: CNTL( 3 )
      INTEGER :: ICNTL( 3 )
      INTEGER :: KEEP( 7 )
      END SUBROUTINE LA15CD
