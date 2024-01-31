! THIS VERSION: GALAHAD 4.3 - 2024-01-06 AT 07:10 GMT.

#include "galahad_modules.h"

!-*-*-  G A L A H A D  -  D U M M Y   M C 6 1    S U B R O U T I N E S  -*-*-

      SUBROUTINE MC61I( ICNTL, CNTL )
      USE GALAHAD_KINDS
      REAL :: CNTL( 5 )
      INTEGER ( KIND = ip_ ) :: ICNTL( 10 )
      ICNTL( 1 ) = 6
      ICNTL( 2 ) = 6
      ICNTL( 3 : 10 ) = 0
      CNTL( 1 : 5 ) = 0.0D+0
      IF ( ICNTL(1) >= 0 ) WRITE(ICNTL(1),
     & "( ' We regret that the solution options that you have ', /,
     &  ' chosen are not all freely available with GALAHAD.', /,
     &  ' If you have HSL (formerly the Harwell Subroutine', /,
     &  ' Library), this option may be enabled by replacing the dummy',
     &   /, ' subroutine MC61I with its HSL namesake ', /,
     &  ' and dependencies. See ', /,
     &  '   $GALAHAD/src/makedefs/packages for details.' )" )
      END SUBROUTINE MC61I

      SUBROUTINE MC61A( job, n, lirn, IRN, ICPTR, PERM, liw,
     &                  IW, W, ICNTL, CNTL, INFO, RINFO )
      USE GALAHAD_KINDS
      USE GALAHAD_SYMBOLS
      INTEGER ( KIND = ip_ ) :: job, n, liw, lirn
      REAL :: RINFO( 15 )
      REAL :: CNTL( 5 ), W( n )
      INTEGER ( KIND = ip_ ) IRN( lirn ), ICPTR( n + 1 ), INFO( 10 )
      INTEGER ( KIND = ip_ ) ICNTL( 10 ), IW( liw ), PERM( n )
      INFO( 1 ) = GALAHAD_unavailable_option
      IF ( ICNTL( 1 ) >= 0 ) WRITE( ICNTL( 1 ),
     & "( ' We regret that the solution options that you have ', /,
     &  ' chosen are not all freely available with GALAHAD.', /,
     &  ' If you have HSL (formerly the Harwell Subroutine', /,
     &  ' Library), this option may be enabled by replacing the dummy',
     &   /, ' subroutine MC61A with its HSL namesake ', /,
     &  ' and dependencies. See ', /,
     &  '   $GALAHAD/src/makedefs/packages for details.' )" )
      END SUBROUTINE MC61A

