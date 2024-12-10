! THIS VERSION: GALAHAD 5.0 - 2024-03-17 AT 11:25 GMT.

#include "hsl_subset.h"

!-*-*-*-*-*-*-  G A L A H A D  -  MA27  S U B R O U T I N E S *-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  February 13th 1995

      SUBROUTINE MA27AR( n, nz, IRN, ICN, IW, liw, IKEEP,
     *                   IW1, nsteps, iflag, ICNTL, CNTL, INFO, ops )

      USE HSL_KINDS_real, ONLY: ip_, rp_

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nz, liw
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iflag
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: nsteps
      REAL ( KIND = rp_ ), INTENT( OUT ) :: ops
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( liw ) :: IW
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n,3 ) :: IKEEP
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n, 2 ) :: IW1
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 20 ) :: INFO
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( 5 ) :: CNTL

!  Dummy subroutine available with GALAHAD

      IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) > 0 )
     *  WRITE ( ICNTL( 2 ), 2000 )
      INFO( 1 ) = - 26
      RETURN

!  Non-executable statements

 2000 FORMAT( /,
     *     ' We regret that the solution options that you have ', /,
     *     ' chosen are not all freely available with GALAHAD.', //,
     *     ' If you have HSL (formerly the Harwell Subroutine',
     *     ' Library), this ', /,
     *     ' option may be enabled by replacing the dummy ', /,
     *     ' subroutine MA27AR with its HSL namesake ', /,
     *     ' and dependencies. See ', /,
     *     '   $GALAHAD/src/makedefs/packages for details.', //,
     *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MA27AR

      END SUBROUTINE MA27AR

      SUBROUTINE MA27BR( n, nz, IRN, ICN, A, la, IW, liw,
     *                   IKEEP, nsteps, maxfrt, IW1, ICNTL, CNTL, INFO )

      USE HSL_KINDS_real, ONLY: ip_, rp_

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nz, la, liw, nsteps
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: maxfrt
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( liw ) :: IW
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n, 3 ) :: IKEEP
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: IW1
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 20 ) :: INFO
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( 5 ) :: CNTL
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( la ) :: A

!  Dummy subroutine available with GALAHAD

      IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) > 0 )
     *  WRITE ( ICNTL( 2 ), 2000 )
      INFO( 1 ) = - 26
      RETURN

!  Non-executable statements

 2000 FORMAT( /,
     *     ' We regret that the solution options that you have ', /,
     *     ' chosen are not all freely available with GALAHAD.', //,
     *     ' If you have HSL (formerly the Harwell Subroutine',
     *     ' Library), this ', /,
     *     ' option may be enabled by replacing the dummy ', /,
     *     ' subroutine MA27BR with its HSL namesake ', /,
     *     ' and dependencies. See ', /,
     *     '   $GALAHAD/src/makedefs/packages for details.', //,
     *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MA27BR

      END SUBROUTINE MA27BR

      SUBROUTINE MA27CR( n, A, la, IW, liw, W, maxfrt, RHS, 
     *                   IW1, nsteps, ICNTL, INFO )

      USE HSL_KINDS_real, ONLY: ip_, rp_

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, la, liw, maxfrt, nsteps
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( liw ) :: IW
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( nsteps ) :: IW1
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 20 ) :: INFO
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( la ) :: A
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( maxfrt ) :: W
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: RHS

!  Dummy subroutine available with GALAHAD

      IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) > 0 )
     *  WRITE ( ICNTL( 2 ), 2000 )
      INFO( 1 ) = - 26
      RETURN

!  Non-executable statements

 2000 FORMAT( /,
     *     ' We regret that the solution options that you have ', /,
     *     ' chosen are not all freely available with GALAHAD.', //,
     *     ' If you have HSL (formerly the Harwell Subroutine',
     *     ' Library), this ', /,
     *     ' option may be enabled by replacing the dummy ', /,
     *     ' subroutine MA27CR with its HSL namesake ', /,
     *     ' and dependencies. See ', /,
     *     '   $GALAHAD/src/makedefs/packages for details.', //,
     *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MA27CR

      END SUBROUTINE MA27CR

      SUBROUTINE MA27IR( ICNTL, CNTL )

      USE HSL_KINDS_real, ONLY: ip_, rp_

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 30 ) :: ICNTL
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( 5 ) :: CNTL

!  Dummy subroutine available with GALAHAD

      ICNTL( 1 ) = - 1
      ICNTL( 4 ) = - 1
!     WRITE ( 6, 2000 )
      RETURN

!  Non-executable statements

!2000 FORMAT( /,
!    *     ' We regret that the solution options that you have ', /,
!    *     ' chosen are not all freely available with GALAHAD.', //,
!    *     ' If you have HSL (formerly the Harwell Subroutine',
!    *     ' Library), this ', /,
!    *     ' option may be enabled by replacing the dummy ', /,
!    *     ' subroutine MA27IR with its HSL namesake ', /,
!    *     ' and dependencies. See ', /,
!    *     '   $GALAHAD/src/makedefs/packages for details.', //,
!    *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MA27IR

      END SUBROUTINE MA27IR

      SUBROUTINE MA27QR( n, A, la, IW, liw, W, maxfnt, RHS,
     *                   IW2, nblk, latop, ICNTL )

      USE HSL_KINDS_real, ONLY: ip_, rp_

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, la, liw, maxfnt, nblk
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: latop
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( nblk ) :: IW2
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( liw ) :: IW
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( la ) :: A
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: RHS
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( maxfnt ) :: W

!  Dummy subroutine available with GALAHAD

      WRITE ( 6, 2000 )
      STOP

!  Non-executable statements

 2000 FORMAT( /,
     *     ' We regret that the solution options that you have ', /,
     *     ' chosen are not all freely available with GALAHAD.', //,
     *     ' If you have HSL (formerly the Harwell Subroutine',
     *     ' Library), this ', /,
     *     ' option may be enabled by replacing the dummy ', /,
     *     ' subroutine MA27QR with its HSL namesake ', /,
     *     ' and dependencies. See ', /,
     *     '   $GALAHAD/src/makedefs/packages for details.', //,
     *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MA27QR

      END SUBROUTINE MA27QR
