! THIS VERSION: GALAHAD 4.1 - 2022-12-10 AT 10:40 GMT.

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

     USE GALAHAD_PRECISION, ONLY : sp_, dp_, long_

     IMPLICIT NONE
 
     PRIVATE
     PUBLIC :: CLOCK_time

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

     REAL ( KIND = sp_ ), INTENT( out ) :: time

!  local variables

     INTEGER ( KIND = long_ ) :: count, count_rate

!  compute the time in seconds

     CALL SYSTEM_CLOCK( count = count, count_rate = count_rate )
     time = REAL( count, KIND = sp_ ) / REAL( count_rate, KIND = sp_ )
     RETURN

!  End of subroutine CLOCK_time_single

     END SUBROUTINE CLOCK_time_single

     SUBROUTINE CLOCK_time_double( time )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!   Provides the current processor clock time in (double-precision) seconds

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  Dummy arguments

     REAL ( KIND = dp_ ), INTENT( out ) :: time

!  local variables

     INTEGER ( KIND = long_ ) :: count, count_rate

!  compute the time in seconds

     CALL SYSTEM_CLOCK( count = count, count_rate = count_rate )
     time = REAL( count, KIND = dp_ ) / REAL( count_rate, KIND = dp_ )
     RETURN

!  End of subroutine CLOCK_time_double

     END SUBROUTINE CLOCK_time_double

!  End of module GALAHAD_CLOCK

   END MODULE GALAHAD_CLOCK
