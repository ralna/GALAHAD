! THIS VERSION: GALAHAD 5.2 - 2025-02-19 AT 10:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-  G A L A H A D   R U N M E T I S _ S I F  *-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 5.2. February 19th 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNMETIS51_SIF_precision

!    -----------------------------------------------------------
!    | Main program for the SIF/CUTEst interface to METIS 5.1, |
!    | a method for ordering symmetric sparse matices          |
!    -----------------------------------------------------------

   USE GALAHAD_KINDS_precision
   USE GALAHAD_USEMETIS51_precision, ONLY: USE_METIS51

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_METIS51( input )

!  Close the data input file

   CLOSE( input  )
   STOP

!  End of RUNMETIS51_SIF_precision

   END PROGRAM RUNMETIS51_SIF_precision
