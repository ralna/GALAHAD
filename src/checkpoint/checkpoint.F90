! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-  G A L A H A D    C H E C K P O I N T   M O D U L E  -*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Daniel Robinson and Nick Gould
!
!  History -
!   originally released GALAHAD Version 2.8. November 16th 2015

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_CHECKPOINT_precision

      USE GALAHAD_KINDS_precision

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: CHECKPOINT

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_

    CONTAINS

!-*-*-*-  G A L A H A D   C H E C K P O I N T    S U B R O U T I N E   -*-*-*-*-

       SUBROUTINE CHECKPOINT( iter, time, measure, checkpointsIter,            &
                              checkpointsTime, low, up )

!  Find the iteration, iter, for which the criterion, measure, is
!  first smaller than 10 ** -i, i = low, ..., up

!  Dummy arguments

       INTEGER ( KIND = ip_ ), INTENT( IN ) :: iter, low, up
       REAL, INTENT( IN ) :: time
       REAL ( KIND = rp_ ), INTENT( IN ) :: measure
       INTEGER ( KIND = ip_ ), INTENT( INOUT ),                                &
                               DIMENSION( low : up ) :: checkpointsIter
       REAL ( KIND = rp_ ), INTENT( INOUT ),                                   &
                            DIMENSION( low : up ) :: checkpointsTime

!  Local variable

       INTEGER ( KIND = ip_ ) :: i

       DO i = low, up
         IF ( checkpointsIter( i ) >= 0 ) CYCLE
         IF ( measure <= ten ** ( - i ) ) THEN
           checkpointsIter( i ) = iter
           checkpointsTime( i ) = REAL( time, KIND = rp_ )
         END IF
       END DO

       RETURN

!  End of subroutine CHECKPOINT

       END SUBROUTINE CHECKPOINT

!  End of module GALAHAD_CHECKPOINT

    END MODULE GALAHAD_CHECKPOINT_precision

