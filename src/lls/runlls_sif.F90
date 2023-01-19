! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N L L S _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Nick Gould and Dominique Orban, for GALAHAD productions
!  Copyright reserved
!  October 20th 2007

   PROGRAM RUNLLS_SIF_precision
   USE GALAHAD_KINDS_precision
   USE GALAHAD_USELLS_precision

!  Main program for the SIF/CUTEr interface to LLS, a conjugate-gradient
!  algorithm for solving linear least-squares problems

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_LLS( input )

!  Close the data input file

   CLOSE( input  )
   STOP

!  End of RUNLLS_SIF_precision

   END PROGRAM RUNLLS_SIF_precision
