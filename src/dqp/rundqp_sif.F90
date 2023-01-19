! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N D Q P _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.5. August 1st 2012

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNDQP_SIF_precision

!    -------------------------------------------------
!    | Main program for the SIF/CUTEr interface to   |
!    | DQP, dual gradient-projection algorithm for   |
!    | convex quadratic & least-distance programming |
!    -------------------------------------------------

   USE GALAHAD_KINDS_precision
   USE GALAHAD_USEDQP_precision

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_DQP( input )

!  Close the data input file

   CLOSE( input  )
   STOP

!  End of RUNDQP_SIF_precision

   END PROGRAM RUNDQP_SIF_precision
