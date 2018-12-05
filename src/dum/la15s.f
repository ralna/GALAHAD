      SUBROUTINE LA15I( ICNTL, CNTL, KEEP )
      REAL :: CNTL( 3 )
      INTEGER :: ICNTL( 3 )
      INTEGER :: KEEP( 7 )
      END SUBROUTINE LA15I

      SUBROUTINE LA15A( A, IND, nzero, ia, n, IP, IW, W, g, u,
     &                   ICNTL, CNTL, KEEP )
      REAL :: g, U
      INTEGER :: ia, n, nzero
      REAL :: A( ia ),W( n )
      INTEGER :: IND( ia, 2 ),IP(N,2),IW( n, 8 )
      REAL :: CNTL( 3 )
      INTEGER :: ICNTL( 3 )
      INTEGER :: KEEP( 7 )
      END SUBROUTINE LA15A

      SUBROUTINE LA15B( A, IND, ia, n, IP, IW, W, g, B,
     &                   trans, ICNTL, KEEP )
      REAL :: g
      INTEGER :: ia, n
      LOGICAL :: trans
      REAL :: A( ia ), B( n ), W( n )
      INTEGER :: IND( ia, 2 ),IP( n, 2 ), IW( n, 4 )
      INTEGER :: ICNTL( 3 )
      INTEGER :: KEEP( 7 )
      END SUBROUTINE LA15B

      SUBROUTINE LA15C( A, IND, ia, n, IP, IW, W, g, u, mm,
     &                   ICNTL, CNTL, KEEP )
      REAL :: g, u
      INTEGER :: ia, mm, n
      REAL :: A( ia ),W( n )
      INTEGER IND( ia, 2 ), IP( n, 2 ), IW( n, 4 )
      REAL :: CNTL( 3 )
      INTEGER :: ICNTL( 3 )
      INTEGER :: KEEP( 7 )
      END SUBROUTINE LA15C
