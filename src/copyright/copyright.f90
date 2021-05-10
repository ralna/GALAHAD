! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-  G A L A H A D    C O P Y R I G H T   M O D U L E  -*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould
!
!  History -
!   originally released pre GALAHAD Version 2.0. May 22nd 2004

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_COPYRIGHT

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: COPYRIGHT

    CONTAINS

!-*-*-*-*-  G A L A H A D   C O P Y R I G H T  S U B R O U T I N E   -*-*-*=*-

      SUBROUTINE COPYRIGHT( out, startyear )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: out
      CHARACTER ( LEN = 4 ), INTENT( IN ) :: startyear

!-----------------------------------------------
!   L o c a l   V a r i b l e s
!-----------------------------------------------

      CHARACTER ( LEN = 8 ) :: currentdate

      CALL date_and_time( date = currentdate )

      IF ( startyear == currentdate( 1 : 4 ) ) THEN
        WRITE( out, "(                                                         &
     &      /, ' Copyright GALAHAD productions, ', A4,                         &
     &     //, ' - Use of this code is restricted to those who agree to abide',&
     &     /,  ' - by the conditions-of-use described in the  README.cou file',&
     &     /,  ' - distributed with the source of the  GALAHAD  codes or from',&
     &     /,  ' - the  WWW  at  http://galahad.rl.ac.uk/galahad-www/cou.html',&
     &     / )" ) currentdate( 1 : 4 )
      ELSE
        WRITE( out, "(                                                         &
     &      /, ' Copyright GALAHAD productions, ', A4, '-', A4                 &
     &     //, ' - Use of this code is restricted to those who agree to abide',&
     &     /,  ' - by the conditions-of-use described in the  README.cou file',&
     &     /,  ' - distributed with the source of the  GALAHAD  codes or from',&
     &     /,  ' - the  WWW  at  http://galahad.rl.ac.uk/galahad-www/cou.html',&
     &     / )" ) startyear, currentdate( 1 : 4 )
      END IF

!  End of subroutine COPYRIGHT

      END SUBROUTINE COPYRIGHT

!  End of module GALAHAD_COPYRIGHT

    END MODULE GALAHAD_COPYRIGHT

