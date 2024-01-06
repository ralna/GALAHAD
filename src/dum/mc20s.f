! THIS VERSION: GALAHAD 4.3 - 2024-01-05 AT 14:40 GMT.

!-*-*-*-*-*-  G A L A H A D  -  D U M M Y   M C 2 0   S U B R O U T I N E *-*-*-

      SUBROUTINE MC20A( nc, maxa, A, INUM, JPTR, JNUM, jdisp )

      USE GALAHAD_KINDS_single

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: nc, maxa, jdisp
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( maxa ) :: INUM
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( maxa ) :: JNUM
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( nc ) :: JPTR
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( maxa ) :: A

!  Dummy subroutine available with GALAHAD

!     WRITE ( 6, 2000 )
      JPTR( 1 ) = - 1
      RETURN

!  Non-executable statements

! 2000 FORMAT( /,
!     *     ' We regret that the solution options that you have ', /,
!     *     ' chosen are not all freely available with GALAHAD.', //,
!     *     ' If you have HSL (formerly the Harwell Subroutine',
!     *     ' Library), this ', /,
!     *     ' option may be enabled by replacing the dummy ', /,
!     *     ' subroutine MC20A with its HSL namesake ', /,
!     *     ' and dependencies. See ', /,
!     *     '   $GALAHAD/src/makedefs/packages for details.', //,
!     *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MC20A

      END SUBROUTINE MC20A
