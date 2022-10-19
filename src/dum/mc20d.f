! THIS VERSION: 2022-10-08 AT 13:20:00 GMT.

!-*-*-*-*-*-  G A L A H A D  -  D U M M Y   M C 2 0   S U B R O U T I N E *-*-*-

      SUBROUTINE MC20AD( nc, maxa, A, INUM, JPTR, JNUM, jdisp )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, INTENT( IN ) :: nc, maxa, jdisp
      INTEGER, INTENT( INOUT ), DIMENSION( maxa ) :: INUM, JNUM
      INTEGER, INTENT( OUT ), DIMENSION( nc ) :: JPTR
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( maxa ) :: A

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
!     *     ' subroutine MC20AD with its HSL namesake ', /,
!     *     ' and dependencies. See ', /,
!     *     '   $GALAHAD/src/makedefs/packages for details.', //,
!     *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MC20AD

      END SUBROUTINE MC20AD
