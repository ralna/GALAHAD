! THIS VERSION: 25/06/2002 AT 14:00:00 PM.
! Updated 25/06/2002: additional warning information added

!-*-*-*-*-*-*-  L A N C E L O T  -B-  DUMICFS  S U B R O U T I N E S *-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  February 13th 1995

!  ---------------------------------------------------------------
!  These subroutine definitions should be replaced by their actual
!  MINPACK2 counterparts an d dependencies if available

      SUBROUTINE DICFS( N, NNZ, A, ADIAG, ACOL_PTR, AROW_IND,
     *                  L, LDIAG, LCOL_PTR, LROW_IND,
     *                  P, ALPHA, IWA, WA1, WA2 )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( INOUT ) :: N
      INTEGER, INTENT( IN ) :: NNZ, P
      INTEGER, INTENT( IN ) :: ACOL_PTR( N + 1 ), AROW_IND( NNZ )
      INTEGER :: LCOL_PTR( N + 1 )
      INTEGER :: LROW_IND( NNZ + N * P )
      INTEGER :: IWA( 3 * N )
      DOUBLE PRECISION, INTENT( INOUT ) :: ALPHA
      DOUBLE PRECISION :: WA1( N ), WA2( N )
      DOUBLE PRECISION, INTENT( IN ) :: A( NNZ ), ADIAG( N )
      DOUBLE PRECISION :: L( NNZ + N * P )
      DOUBLE PRECISION :: LDIAG( N )

!  Dummy subroutine available with LANCELOT B

      IF ( IWA( 1 ) >= 0 ) WRITE ( IWA( 1 ), 2000 )
      N = - 26
      RETURN

!  Non-executable statements

 2000 FORMAT( /,
     *    ' We regret that the solution options that you have', /,
     *    ' chosen are not all freely available with LANCELOT.', /,
     *    ' This code is part of the ICFS package (itself part', /,
     *    " of MINPACK2), and may be obtained from Jorge More'",/,
     *    ' (more@mcs.anl.gov) from the WWW page:', /,
     *    '   http:////www-unix.mcs.anl.gov//~more//icfs//', /,
     *    ' If you have the ICFS package, this option may be ', /,
     *    ' enabled by replacing all the dummy subroutines in',/,
     *    ' the LANCELOT dumicfs.f90 file with their MINPACK2', /,
     *    ' namesakes and dependencies, and recompiling it.', /,
     *    ' See $GALAHAD/src/makedefs/packages for details.', //,
     *    ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine DICFS

      END SUBROUTINE DICFS

      SUBROUTINE DSTRSOL( N, L, LDIAG, JPTR, INDR, R, TASK )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      CHARACTER ( LEN = 60 ), INTENT( IN ) :: TASK
      INTEGER, INTENT( INOUT ) :: N
      INTEGER, INTENT( IN ) :: JPTR( N + 1 ), INDR( * )
      DOUBLE PRECISION, INTENT( IN ) :: L( * ), LDIAG( N )
      DOUBLE PRECISION, INTENT( INOUT ) :: R( N )

!  Dummy subroutine available with LANCELOT B

      N = - 26
      RETURN

!  End of dummy subroutine DICFS

      END SUBROUTINE  DSTRSOL

