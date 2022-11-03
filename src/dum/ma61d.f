! THIS VERSION: 2022-11-03 AT 08:30 GMT.
! Updated 25/06/2002: additional warning information added

!-*-*-*-*-*-*-  L A N C E L O T  -B-  MA61  S U B R O U T I N E S *-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  November 14th 2001

      SUBROUTINE MA61ID( ICNTL, CNTL, KEEP )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( OUT ) :: ICNTL( 5 ), KEEP( 12 )
      REAL ( KIND = KIND( 1.0D0 ) ), INTENT( OUT ) :: CNTL( 3 )

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
!    *     ' subroutines MA61ID/MA61DD with their HSL namesakes ', /, 
!    *     ' and dependencies. See ', /,
!    *     '   $GALAHAD/src/makedefs/packages for details.', //,
!    *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MA61ID

      END SUBROUTINE MA61ID

      SUBROUTINE MA61DD( A, IRN, ia, n, IK, IP, row, ncp, nucl, nual )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: ia, n
      INTEGER, INTENT( INOUT ) :: ncp, nucl, nual
      LOGICAL, INTENT( IN ) :: row
      INTEGER, INTENT( INOUT ), DIMENSION( ia ) :: IRN
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: IK
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: IP
      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ), 
     *                                DIMENSION( ia ) :: A

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
     *     ' subroutines MA61ID/MA61DD with their HSL namesakes ', /, 
     *     ' and dependencies. See ', /,
     *     '   $GALAHAD/src/makedefs/packages for details.', //,
     *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MA61DD

      END SUBROUTINE MA61DD
