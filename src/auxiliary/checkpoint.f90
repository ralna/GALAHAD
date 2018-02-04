! THIS VERSION: GALAHAD 2.8 - 16/11/2015 AT 11:30 GMT.

!-*-*-*-*-*-  G A L A H A D    N O R M S   M O D U L E  -*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould
!
!  History -
!   originally released GALAHAD Version 2.8. November 16th 2015

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_CHECKPOINT_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: CHECKPOINT

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp

    CONTAINS

!-*-*-*-  G A L A H A D   C H E C K P O I N T    S U B R O U T I N E   -*-*-*-

       SUBROUTINE CHECKPOINT( iter, time, measure, checkpointsIter,            &
                              checkpointsTime, low, up )
       !SUBROUTINE CHECKPOINT( iter, measure, checkpoints, low, up )

!  Find the iteration, iter, for which the criterion, measure, is
!  first smaller than 10 ** -i, i = low, ..., up

!  Dummy arguments

       INTEGER, INTENT( IN ) :: iter, low, up
       REAL, INTENT( IN ) :: time
       REAL ( KIND = wp ), INTENT( IN ) :: measure
       !INTEGER, INTENT( INOUT ), DIMENSION( low : up ) :: checkpoints
       INTEGER, INTENT( INOUT ), DIMENSION( low : up ) :: checkpointsIter
       REAL ( KIND = wp ), INTENT( INOUT ),                                    &
         DIMENSION( low : up ) :: checkpointsTime

!  Local variable

       INTEGER :: i

       DO i = low, up
         !IF ( checkpoints( i ) >= 0 ) CYCLE
         IF ( checkpointsIter( i ) >= 0 ) CYCLE
         IF ( measure <= ten ** ( - i ) ) THEN
            !checkpoints( i ) = iter
            checkpointsIter( i ) = iter
            checkpointsTime( i ) = DBLE(time)
         END IF
       END DO

       RETURN

!  End of subroutine CHECKPOINT

       END SUBROUTINE CHECKPOINT

!  End of module GALAHAD_CHECKPOINT_double

    END MODULE GALAHAD_CHECKPOINT_double

