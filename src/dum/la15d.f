! THIS VERSION: GALAHAD 4.3 - 2024-01-04 AT 14:30 GMT.

#ifdef GALAHAD_64BIT_INTEGER
  INTEGER, PARAMETER :: ip_ = INT64
#else
  INTEGER, PARAMETER :: ip_ = INT32
#endif

      SUBROUTINE LA15ID( ICNTL, CNTL, KEEP )
      DOUBLE PRECISION :: CNTL( 3 )
      INTEGER ( KIND = ip_ ) :: ICNTL( 3 )
      INTEGER ( KIND = ip_ ) :: KEEP( 7 )
      END SUBROUTINE LA15ID

      SUBROUTINE LA15AD( A, IND, nzero, ia, n, IP, IW, W, g, u,
     &                   ICNTL, CNTL, KEEP )
      DOUBLE PRECISION :: g, U
      INTEGER ( KIND = ip_ ) :: ia, n, nzero
      DOUBLE PRECISION :: A( ia ),W( n )
      INTEGER ( KIND = ip_ ) :: IND( ia, 2 ),IP(N,2),IW( n, 8 )
      DOUBLE PRECISION :: CNTL( 3 )
      INTEGER ( KIND = ip_ ) :: ICNTL( 3 )
      INTEGER ( KIND = ip_ ) :: KEEP( 7 )
      END SUBROUTINE LA15AD

      SUBROUTINE LA15BD( A, IND, ia, n, IP, IW, W, g, B,
     &                   trans, ICNTL, KEEP )
      DOUBLE PRECISION :: g
      INTEGER ( KIND = ip_ ) :: ia, n
      LOGICAL :: trans
      DOUBLE PRECISION :: A( ia ), B( n ), W( n )
      INTEGER ( KIND = ip_ ) :: IND( ia, 2 ),IP( n, 2 ), IW( n, 4 )
      INTEGER ( KIND = ip_ ) :: ICNTL( 3 )
      INTEGER ( KIND = ip_ ) :: KEEP( 7 )
      END SUBROUTINE LA15BD

      SUBROUTINE LA15CD( A, IND, ia, n, IP, IW, W, g, u, mm,
     &                   ICNTL, CNTL, KEEP )
      DOUBLE PRECISION :: g, u
      INTEGER ( KIND = ip_ ) :: ia, mm, n
      DOUBLE PRECISION :: A( ia ),W( n )
      INTEGER ( KIND = ip_ ) :: IND( ia, 2 ), IP( n, 2 ), IW( n, 4 )
      DOUBLE PRECISION :: CNTL( 3 )
      INTEGER ( KIND = ip_ ) :: ICNTL( 3 )
      INTEGER ( KIND = ip_ ) :: KEEP( 7 )
      END SUBROUTINE LA15CD
