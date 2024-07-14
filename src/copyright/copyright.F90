! THIS VERSION: GALAHAD 5.0 - 2024-07-13 AT 10:15 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-  G A L A H A D    C O P Y R I G H T   M O D U L E  -*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould
!
!  History -
!   originally released pre GALAHAD Version 2.0. May 22nd 2004

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_COPYRIGHT

      USE GALAHAD_KINDS_precision

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: COPYRIGHT

    CONTAINS

!-*-*-*-*-  G A L A H A D   C O P Y R I G H T  S U B R O U T I N E   -*-*-*=*-

      SUBROUTINE COPYRIGHT( out, startyear )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: out
      CHARACTER ( LEN = 4 ), INTENT( IN ) :: startyear

!-----------------------------------------------
!   L o c a l   V a r i b l e s
!-----------------------------------------------

      CHARACTER ( LEN = 8 ) :: currentdate

      CALL date_and_time( date = currentdate )

      IF ( startyear == currentdate( 1 : 4 ) ) THEN
        WRITE( out, 2000 ) currentdate( 1 : 4 )
      ELSE
        WRITE( out, 2010 ) startyear, currentdate( 1 : 4 )
      END IF

!  non-executable statements

2000 FORMAT( /, ' Copyright GALAHAD productions, ', A4,                        &
           //, ' - Use of this code is restricted to those who agree to abide',&
           /,  ' - by the conditions-of-use described in the BSD LICENSE file',&
           /,  ' - distributed with the source of the  GALAHAD  codes or from',&
           /,  ' - the  WWW  at  http://galahad.rl.ac.uk/download',            &
           / )
2010 FORMAT( /, ' Copyright GALAHAD productions, ', A4, '-', A4                &
           //, ' - Use of this code is restricted to those who agree to abide',&
           /,  ' - by the conditions-of-use described in the BSD LICENSE file',&
           /,  ' - distributed with the source of the  GALAHAD  codes or from',&
           /,  ' - the  WWW  at  http://galahad.rl.ac.uk/download',            &
           / )


!  End of subroutine COPYRIGHT

      END SUBROUTINE COPYRIGHT

!  End of module GALAHAD_COPYRIGHT

    END MODULE GALAHAD_COPYRIGHT

