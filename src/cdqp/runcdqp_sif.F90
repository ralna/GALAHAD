! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N C D Q P _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released as runccqp_sif with GALAHAD Version 2.7, July 17th 2015
!   renamed runcdqp_sif GALAHAD Version 4.1, May 17th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNCDQP_SIF_precision

!     -------------------------------------------------------------------
!    | Main program for the SIF/CUTEr interface to CDQP, an infeasible   |
!    |      primal-dual interior-point to dual gradient-projection       |
!    |         crossover method for convex quadratic programming         |
!     -------------------------------------------------------------------

   USE GALAHAD_KINDS_precision
   USE GALAHAD_USECDQP_precision

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_CDQP( input )

!  Close the data input file

   CLOSE( input  )
   STOP

!  End of RUNCDQP_SIF_precision

   END PROGRAM RUNCDQP_SIF_precision
