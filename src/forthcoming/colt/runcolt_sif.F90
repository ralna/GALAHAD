! THIS VERSION: GALAHAD 4.2 - 2023-10-13 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   R U N C O L T _ S I F  *-*-*-*-*-*-*-*-*-

!  Jessica Farmer, Jaroslav Fowkes and Nick Gould, for GALAHAD productions
!  Copyright reserved
!  October 11th 2023

   PROGRAM RUNCOLT_SIF_precision
   USE GALAHAD_KINDS_precision
   USE GALAHAD_USECOLT_precision

!  Main program for the SIF interface to COLT, a target least-squares
!  algorithm for nonlinear programming

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

   CALL USE_COLT( insif )

!  Close the data input file

   CLOSE( insif  )
   STOP

!  End of RUNCOLT_SIF_precision

   END PROGRAM RUNCOLT_SIF_precision
