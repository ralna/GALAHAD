! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-  G A L A H A D   R U N F I L T R A N E _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  June 9th 2003

   PROGRAM RUNFILTRANE_SIF_precision
   USE GALAHAD_KINDS_precision
   USE GALAHAD_USEFILTRANE_precision

!  Main program for the SIF/CUTEr interface to FILTRANE, a filter
!  trust-region method for feasibility problems.

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: errout = 6
   INTEGER ( KIND = ip_ ), PARAMETER :: insif  = 56  ! OUTSDIF.d device number
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

!  Call the CUTEr interface

   CALL USE_FILTRANE( insif )

!  Close the data input file

   CLOSE( insif  )
   STOP

!  End of RUNFILTRANE_SIF_precision

   END PROGRAM RUNFILTRANE_SIF_precision
