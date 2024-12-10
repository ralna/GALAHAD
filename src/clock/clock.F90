! THIS VERSION: GALAHAD 5.1 - 2024-11-23 AT 15:20 GMT.

#include "galahad_modules.h"

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

#ifdef REAL_128
     USE GALAHAD_KINDS, ONLY : long_, r4_, r8_, r16_
#else
     USE GALAHAD_KINDS, ONLY : long_, r4_, r8_
#endif

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: CLOCK_time

!-------------------------------
!   I n t e r f a c e  B l o c k
!-------------------------------

     INTERFACE CLOCK_time
#ifdef REAL_128
       MODULE PROCEDURE CLOCK_time_single, CLOCK_time_double,                  &
                        CLOCK_time_quadruple
#else
       MODULE PROCEDURE CLOCK_time_single, CLOCK_time_double
#endif
     END INTERFACE CLOCK_time

   CONTAINS

     SUBROUTINE CLOCK_time_single( time )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!   Provides the current processor clock time in (single-precision) seconds

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  Dummy arguments

     REAL ( KIND = r4_ ), INTENT( out ) :: time

!  local variables

     INTEGER ( KIND = long_ ) :: count, count_rate

!  compute the time in seconds

     CALL SYSTEM_CLOCK( count = count, count_rate = count_rate )
     time = REAL( count, KIND = r4_ ) / REAL( count_rate, KIND = r4_ )
     RETURN

!  End of subroutine CLOCK_time_single

     END SUBROUTINE CLOCK_time_single

     SUBROUTINE CLOCK_time_double( time )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!   Provides the current processor clock time in (double-precision) seconds

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  Dummy arguments

     REAL ( KIND = r8_ ), INTENT( out ) :: time

!  local variables

     INTEGER ( KIND = long_ ) :: count, count_rate

!  compute the time in seconds

     CALL SYSTEM_CLOCK( count = count, count_rate = count_rate )
     time = REAL( count, KIND = r8_ ) / REAL( count_rate, KIND = r8_ )
     RETURN

!  End of subroutine CLOCK_time_double

     END SUBROUTINE CLOCK_time_double

#ifdef REAL_128
     SUBROUTINE CLOCK_time_quadruple( time )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!   Provides the current processor clock time in (quadruple-precision) seconds

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!  Dummy arguments

     REAL ( KIND = r16_ ), INTENT( out ) :: time

!  local variables

     INTEGER ( KIND = long_ ) :: count, count_rate

!  compute the time in seconds

     CALL SYSTEM_CLOCK( count = count, count_rate = count_rate )
     time = REAL( count, KIND = r16_ ) / REAL( count_rate, KIND = r16_ )
     RETURN

!  End of subroutine CLOCK_time_quadruple

     END SUBROUTINE CLOCK_time_quadruple
#endif

!  End of module GALAHAD_CLOCK

   END MODULE GALAHAD_CLOCK
