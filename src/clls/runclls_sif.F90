! THIS VERSION: GALAHAD 4.1 - 2022-12-30 AT 09:40 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N C L L S _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 4.1, July 20th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNCLLS_SIF

!    ---------------------------------------------------
!    | Main program for the SIF/CUTEst interface to    |
!    | CLLS, an interior-point crossover algorithm for |
!    | constrained linear least-squares optimization   |
!    ---------------------------------------------------

   USE GALAHAD_KINDS
   USE GALAHAD_USECLLS_precision

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEst interface

   CALL USE_CLLS( input, close_input = .TRUE. )
   STOP

!  End of RUNCLLS_SIF

   END PROGRAM RUNCLLS_SIF
