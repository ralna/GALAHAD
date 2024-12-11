! THIS VERSION: GALAHAD 5.1 - 2024-12-11 AT 08:15 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D _ E R R O R   P R O G R A M  *-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould

!  History -
!   originally released GALAHAD Version 3.3. July 7th 2021

!  Return a description of a GALAHAD status error code

   PROGRAM GALAHAD_error
   USE GALAHAD_SYMBOLS
   USE GALAHAD_KINDS, ONLY : ip_

   INTEGER ( KIND = ip_ ), PARAMETER :: in = 5
   INTEGER ( KIND = ip_ ), PARAMETER :: out = 6
   INTEGER ( KIND = ip_ ) :: status
   READ( in, * ) status
   WRITE( out, "( ' GALAHAD status value ', I0, ' means:' )" )    &
     status
   CALL SYMBOLS_status( status, out, '', 'GALAHAD' )
   STOP
   END PROGRAM GALAHAD_error
