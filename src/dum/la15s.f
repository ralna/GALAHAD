#include "galahad_modules.h"

! THIS VERSION: GALAHAD 4.3 - 2024-01-24 AT 15:50 GMT.

      SUBROUTINE LA15I( ICNTL, CNTL, KEEP )
      USE GALAHAD_KINDS
      REAL :: CNTL( 3 )
      INTEGER ( KIND = ip_ ) :: ICNTL( 3 )
      INTEGER ( KIND = ip_ ) :: KEEP( 7 )
      END SUBROUTINE LA15I

      SUBROUTINE LA15A( A, IND, nzero, ia, n, IP, IW, W, g, u,
     &                  ICNTL, CNTL, KEEP )
      USE GALAHAD_KINDS
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
      USE GALAHAD_KINDS
      REAL :: g
      INTEGER ( KIND = ip_ ) :: ia, n
      LOGICAL ( KIND = lp_ ) :: trans
      REAL :: A( ia ), B( n ), W( n )
      INTEGER ( KIND = ip_ ) :: IND( ia, 2 ),IP( n, 2 ), IW( n, 4 )
      INTEGER ( KIND = ip_ ) :: ICNTL( 3 )
      INTEGER ( KIND = ip_ ) :: KEEP( 7 )
      END SUBROUTINE LA15B

      SUBROUTINE LA15C( A, IND, ia, n, IP, IW, W, g, u, mm,
     &                  ICNTL, CNTL, KEEP )
      USE GALAHAD_KINDS
      REAL :: g, u
      INTEGER ( KIND = ip_ ) :: ia, mm, n
      REAL :: A( ia ),W( n )
      INTEGER ( KIND = ip_ ) :: IND( ia, 2 ), IP( n, 2 ), IW( n, 4 )
      REAL :: CNTL( 3 )
      INTEGER ( KIND = ip_ ) :: ICNTL( 3 )
      INTEGER ( KIND = ip_ ) :: KEEP( 7 )
      END SUBROUTINE LA15C
