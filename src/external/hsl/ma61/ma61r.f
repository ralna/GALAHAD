! THIS VERSION: GALAHAD 5.0 - 2024-03-17 AT 11:25 GMT.

#include "hsl_subset.h"

!-*-*-*-*-*-*-  L A N C E L O T  -B-  M A 6 1  S U B R O U T I N E S *-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  November 14th 2001

      SUBROUTINE MA61IR( ICNTL, CNTL, KEEP )

      USE HSL_KINDS_real, ONLY: ip_, rp_

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER(ip_), INTENT( OUT ) :: ICNTL( 5 ), KEEP( 12 )
      REAL(rp_), INTENT( OUT ) :: CNTL( 3 )

!  Dummy subroutine available with LANCELOT

      ICNTL( 1 ) = - 1
      RETURN
!     WRITE ( 6, 2000 )
!     STOP

!  Non-executable statements

!2000 FORMAT( /,
!    *     ' We regret that the solution options that you have ', /,
!    *     ' chosen are not all freely available with LANCELOT B.', //,
!    *     ' If you have HSL (formerly the Harwell Subroutine',
!    *     ' Library), this ', /,
!    *     ' option may be enabled by replacing the dummy ', /,
!    *     ' subroutines MA61IR/MA61DR with their HSL namesakes ', /,
!    *     ' and dependencies. See ', /,
!    *     '   $GALAHAD/src/makedefs/packages for details.', //,
!    *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MA61IR

      END SUBROUTINE MA61IR

      SUBROUTINE MA61DR( A, IRN, ia, n, IK, IP, row, ncp,
     *                   nucl, nual )

      USE HSL_KINDS_real, ONLY: ip_, rp_, lp_

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER(ip_), INTENT( IN ) :: ia, n
      INTEGER(ip_), INTENT( INOUT ) :: ncp, nucl, nual
      LOGICAL(lp_), INTENT( IN ) :: row
      INTEGER(ip_), INTENT( INOUT ), DIMENSION( ia ) :: IRN
      INTEGER(ip_), INTENT( INOUT ), DIMENSION( n ) :: IK
      INTEGER(ip_), INTENT( INOUT ), DIMENSION( n ) :: IP
      REAL(rp_), INTENT( INOUT ), DIMENSION( ia ) :: A

!  Dummy subroutine available with LANCELOT

      WRITE ( 6, 2000 )
      STOP

!  Non-executable statements

 2000 FORMAT( /,
     *     ' We regret that the solution options that you have ', /,
     *     ' chosen are not all freely available with LANCELOT B.', //,
     *     ' If you have HSL (formerly the Harwell Subroutine',
     *     ' Library), this ', /,
     *     ' option may be enabled by replacing the dummy ', /,
     *     ' subroutines MA61IR/MA61DR with their HSL namesakes ', /,
     *     ' and dependencies. See ', /,
     *     '   $GALAHAD/src/makedefs/packages for details.', //,
     *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MA61DR

      END SUBROUTINE MA61DR
