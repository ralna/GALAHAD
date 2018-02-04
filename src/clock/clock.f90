! THIS VERSION: GALAHAD 2.5 - 11/05/2011 AT 15:30 GMT.

!-*-*-*-*-*-*-*-*- G A L A H A D _ C L O C K    M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. January 27th 2011

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_CLOCK

!   ----------------------------------
!   | Provides the system clock time |
!   ----------------------------------

     IMPLICIT NONE
 
     PRIVATE
     PUBLIC :: CLOCK_time

!----------------------
!   P r e c i s i o n s
!----------------------

     INTEGER, PARAMETER :: sp = KIND( 1.0 )
     INTEGER, PARAMETER :: dp = KIND( 1.0D+0 )

!-------------------------------
!   I n t e r f a c e  B l o c k
!-------------------------------

     INTERFACE CLOCK_time
       MODULE PROCEDURE CLOCK_time_single, CLOCK_time_double
     END INTERFACE CLOCK_time

   CONTAINS

     SUBROUTINE CLOCK_time_single( time )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!   Provides the current processor clock time in (single-precision) seconds

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  Dummy arguments

     REAL ( KIND = sp ), INTENT( out ) :: time

!  local variables

     INTEGER :: count, count_rate

!  compute the time in seconds

     CALL SYSTEM_CLOCK( count = count, count_rate = count_rate )
     time = REAL( count, KIND = sp ) / REAL( count_rate, KIND = sp )
     RETURN

!  End of subroutine CLOCK_time_single

     END SUBROUTINE CLOCK_time_single

     SUBROUTINE CLOCK_time_double( time )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!   Provides the current processor clock time in (double-precision) seconds

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  Dummy arguments

     REAL ( KIND = dp ), INTENT( out ) :: time

!  local variables

     INTEGER :: count, count_rate

!  compute the time in seconds

     CALL SYSTEM_CLOCK( count = count, count_rate = count_rate )
     time = REAL( count, KIND = dp ) / REAL( count_rate, KIND = dp )
     RETURN

!  End of subroutine CLOCK_time_double

     END SUBROUTINE CLOCK_time_double

!  End of module GALAHAD_CLOCK

   END MODULE GALAHAD_CLOCK
