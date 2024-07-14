! THIS VERSION: GALAHAD 5.1 - 2024-07-14 AT 14:00 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N B N L S _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  October 27th 2015

   PROGRAM RUNBNLS_SIF_precision
   USE GALAHAD_KINDS_precision
   USE GALAHAD_USEBNLS_precision

!  Main program for the SIF interface to BNLS, a regularization algorithm for
!  nonlinear least-squares optimization

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

   CALL USE_BNLS( insif )

!  Close the data input file

   CLOSE( insif )
   STOP

!  End of RUNBNLS_SIF_precision

   END PROGRAM RUNBNLS_SIF_precision
