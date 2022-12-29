! THIS VERSION: GALAHAD 4.1 - 2022-12-29 AT 15:00 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-  G A L A H A D   R U N P Q P _ S I F  *-*-*-*-*-*-

!  Nick Gould and Dominique Orban, for GALAHAD productions
!  Copyright reserved
!  September 14th 2004

   PROGRAM RUNPQP_SIF
   USE GALAHAD_PRECISION
   USE GALAHAD_USEPQP_precision

!  Main program for the SIF/CUTEr interface to QPA, a working-set 
!  algorithm for solving quadratic programs

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_PQP( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNPQP_SIF

   END PROGRAM RUNPQP_SIF
