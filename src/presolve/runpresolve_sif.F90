! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-  G A L A H A D   R U N P R E S O L V E _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  March 14th 2003

   PROGRAM RUNPRESOLVE_SIF_precision
   USE GALAHAD_KINDS_precision
   USE GALAHAD_USEPRESOLVE_precision

!  Main program for the SIF/CUTEr interface to PRESOLVE, a preprocessing
!  algorithm for quadratic programs

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_PRESOLVE( input )

!  Close the data input file

   CLOSE( input  )
   STOP

!  End of RUNPRESOLVE_SIF_precision

   END PROGRAM RUNPRESOLVE_SIF_precision
