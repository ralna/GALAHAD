! THIS VERSION: GALAHAD 5.0 - 2024-03-17 AT 11:25 GMT.

#include "hsl_subset.h"

!-*-*-*-*-*-*-  G A L A H A D  -  L A 0 4  S U B R O U T I N E S *-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  October 19th 2018

      SUBROUTINE LA04AR( A, la, IRN, IP, m, n, B, C, BND,
     &                   kb, lb, job, CNTL, IX, JX, X, Z,
     &                   G, RINFO, WS, lws, IWS, liws )

      USE HSL_KINDS_real, ONLY: ip_, rp_

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: la, m, n, kb, lb
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: lws, liws
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: job
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n + 1 ) :: IP
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( la ) :: IRN
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( m ) :: IX
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( kb ) :: JX
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( liws ) :: IWS
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( la ) :: A
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( m ) :: B
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( 2, kb ) :: BND
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: C
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( 15 ) :: CNTL
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( 40 ) :: RINFO
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n + m ) :: X
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: Z
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: G
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( lws ) :: WS

!  Dummy subroutine available with GALAHAD

!     WRITE ( 6, 2000 )
      job = - 101
      RETURN

!  Non-executable statements

!2000 FORMAT( /,
!    &     ' We regret that the solution options that you have ', /,
!    &     ' chosen are not all freely available with GALAHAD.', //,
!    &     ' If you have HSL (formerly the Harwell Subroutine',
!    &     ' Library), ', /,
!    &     ' this option may be enabled by replacing the dummy ', /,
!    &     ' subroutine LA04AR with its HSL namesake ', /,
!    &     ' and dependencies. See ', /,
!    &     '   $GALAHAD/src/makedefs/packages for details.', //,
!    &     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine LA04AR

      END SUBROUTINE LA04AR
