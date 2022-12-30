! THIS VERSION: GALAHAD 4.1 - 2022-12-30 AT 09:40 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N C C Q P _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released as runcqp_sif with GALAHAD Version 2.4, January 1st 2010
!   modified to form runccqp_sif with GALAHAD 4.1, May 17th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNCCQP_SIF

!    ---------------------------------------------------
!    | Main program for the SIF/CUTEst interface to    |
!    | CCQP, an interior-point crossover algorithm for |
!    | convex quadratic & least-distance programming   |
!    ---------------------------------------------------

   USE GALAHAD_KINDS
   USE GALAHAD_USECCQP_precision

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEst interface

   CALL USE_CCQP( input, close_input = .TRUE. )
   STOP

!  End of RUNCCQP_SIF

   END PROGRAM RUNCCQP_SIF
