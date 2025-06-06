! THIS VERSION: GALAHAD 5.1 - 2024-10-28 AT 08:00 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-  G A L A H A D   R U N Q P F _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  June 25th 2012

   PROGRAM RUNQPF_SIF_precision
   USE GALAHAD_KINDS_precision
   USE GALAHAD_USEQPF_precision

!  Main program for the SIF interface to QPF, a quadratic penalty function
!  algorithm for constrained optimization

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

   CALL USE_QPF( insif )

!  Close the data input file

   CLOSE( insif )
   STOP

!  End of RUNQPF_SIF_precision

   END PROGRAM RUNQPF_SIF_precision
