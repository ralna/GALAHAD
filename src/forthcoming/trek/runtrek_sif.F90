! THIS VERSION: GALAHAD 5.3 - 2025-05-20 AT 10:05 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-  G A L A H A D   R U N T R E K _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  May 20th 2025

   PROGRAM RUNTREK_SIF_precision
   USE GALAHAD_KINDS_precision
   USE GALAHAD_USETREK_precision

!  Main program for the SIF interface to TREK, a solver for the trekst-region
!  subproblem

!  Problem insif characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: errout = 6
   INTEGER ( KIND = ip_ ), PARAMETER :: insif = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'
   INTEGER ( KIND = ip_ ) :: iostat

!  Open the data input file

   OPEN( insif, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD',             &
         IOSTAT = iostat )
   IF ( iostat > 0 ) THEN
     WRITE( errout,                                                            &
       "( ' ERROR: could not open file OUTSDIF.d on unit ', I2 )" ) insif
     STOP
   END IF
   REWIND insif

!  Call the SIF interface

   CALL USE_TREK( insif )

!  Close the data input file

   CLOSE( insif )
   STOP

!  End of RUNTREK_SIF_precision

   END PROGRAM RUNTREK_SIF_precision
