! THIS VERSION: GALAHAD 4.1 - 2022-12-22 AT 10:00 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N W A R M _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.5. March 10th 2013

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNWARM_SIF

!    ----------------------------------------------------------
!    | Main program for the SIF/CUTEst interface to WARMSTART |
!    | a a test of the warmstart capabilities of the GALAHAD  |
!    | qp solver DQP                                          |
!    ----------------------------------------------------------

   USE GALAHAD_PRECISION
   USE GALAHAD_usewarm_precision

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEst interface

   CALL USE_warm( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNWARM_SIF

   END PROGRAM RUNWARM_SIF
