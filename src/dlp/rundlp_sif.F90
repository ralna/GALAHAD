! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N D L P _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.6. January 30th 2015

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNDLP_SIF_precision

!     ------------------------------------------
!    | Main program for the SIF/CUTEr interface |
!    | to DLP, dual gradient-projection         |
!    | algorithm for linear programming         |
!     ------------------------------------------

   USE GALAHAD_KINDS_precision
   USE GALAHAD_USEDLP_precision

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_DLP( input )

!  Close the data input file

   CLOSE( input  )
   STOP

!  End of RUNDLP_SIF_precision

   END PROGRAM RUNDLP_SIF_precision
