! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N C Q P S _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.4. January 1st 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNCQPS_SIF_precision

!    -----------------------------------------------------------
!    | Main program for the SIF/CUTEr interface to CQPS,       |
!    | a preconditioned projected conjugate-gradient BQP-based |
!    | method, based on Spelucci's bound-constrained exact     |
!    | penalty reformuation, for convex quadratic programming  |
!    -----------------------------------------------------------

   USE GALAHAD_KINDS_precision
   USE GALAHAD_USECQPS_precision

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_CQPS( input )

!  Close the data input file

   CLOSE( input  )
   STOP

!  End of RUNCQPS_SIF_precision

   END PROGRAM RUNCQPS_SIF_precision
