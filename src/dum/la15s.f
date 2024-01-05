! THIS VERSION: GALAHAD 4.3 - 2024-01-04 AT 14:30 GMT.

#ifdef GALAHAD_64BIT_INTEGER
  INTEGER, PARAMETER :: ip_ = INT64
#else
  INTEGER, PARAMETER :: ip_ = INT32
#endif

      SUBROUTINE LA15I( ICNTL, CNTL, KEEP )
      REAL :: CNTL( 3 )
      INTEGER ( KIND = ip_ ) :: ICNTL( 3 )
      INTEGER ( KIND = ip_ ) :: KEEP( 7 )
      END SUBROUTINE LA15I

      SUBROUTINE LA15A( A, IND, nzero, ia, n, IP, IW, W, g, u,
     &                  ICNTL, CNTL, KEEP )
      REAL :: g, U
      INTEGER ( KIND = ip_ ) :: ia, n, nzero
      REAL :: A( ia ),W( n )
      INTEGER ( KIND = ip_ ) :: IND( ia, 2 ),IP(N,2),IW( n, 8 )
      REAL :: CNTL( 3 )
      INTEGER ( KIND = ip_ ) :: ICNTL( 3 )
      INTEGER ( KIND = ip_ ) :: KEEP( 7 )
      END SUBROUTINE LA15A

      SUBROUTINE LA15B( A, IND, ia, n, IP, IW, W, g, B,
     &                  trans, ICNTL, KEEP )
      REAL :: g
      INTEGER ( KIND = ip_ ) :: ia, n
      LOGICAL ( KIND = ip_ ) :: trans
      REAL :: A( ia ), B( n ), W( n )
      INTEGER ( KIND = ip_ ) :: IND( ia, 2 ),IP( n, 2 ), IW( n, 4 )
      INTEGER ( KIND = ip_ ) :: ICNTL( 3 )
      INTEGER ( KIND = ip_ ) :: KEEP( 7 )
      END SUBROUTINE LA15B

      SUBROUTINE LA15C( A, IND, ia, n, IP, IW, W, g, u, mm,
     &                  ICNTL, CNTL, KEEP )
      REAL :: g, u
      INTEGER ( KIND = ip_ ) :: ia, mm, n
      REAL :: A( ia ),W( n )
      INTEGER ( KIND = ip_ ) :: IND( ia, 2 ), IP( n, 2 ), IW( n, 4 )
      REAL :: CNTL( 3 )
      INTEGER ( KIND = ip_ ) :: ICNTL( 3 )
      INTEGER ( KIND = ip_ ) :: KEEP( 7 )
      END SUBROUTINE LA15C
