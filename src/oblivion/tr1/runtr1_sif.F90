! THIS VERSION: GALAHAD 4.1 - 2023-01-29 AT 14:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-  G A L A H A D   R U N T R 1 _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  January 29th 2023

   PROGRAM RUNTR1_SIF_precision
   USE GALAHAD_KINDS_precision
   USE GALAHAD_USETR1_precision

!  Main program for the SIF interface to TR1, a first-order (steepest-descent)
!  trust-region algorithm for unconstrained optimization

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

   CALL USE_TR1( insif )

!  Close the data input file

   CLOSE( insif )
   STOP

!  End of RUNTR1_SIF_precision

   END PROGRAM RUNTR1_SIF_precision
