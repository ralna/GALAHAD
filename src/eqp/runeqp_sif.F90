! THIS VERSION: GALAHAD 4.1 - 2022-12-30 AT 09:40 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N E Q P _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Nick Gould and Dominique Orban, for GALAHAD productions
!  Copyright reserved
!  March 25th 2004

   PROGRAM RUNEQP_SIF
   USE GALAHAD_KINDS
   USE GALAHAD_USEEQP_precision

!  Main program for the SIF/CUTEr interface to EQP, a projected
!  conjugate-gradient algorithm for solving equality-constrained 
!  quadratic programs

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_EQP( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNEQP_SIF

   END PROGRAM RUNEQP_SIF
