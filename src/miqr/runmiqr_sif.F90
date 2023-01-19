! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N M I Q R _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.6. May 23rd 2014

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNMIQR_SIF_precision

!     -------------------------------------------
!    | Main program for the SIF/CUTEst interface |
!    | to MIQR, a multilevel incomplete QR       |
!    | factorization algorithm by Ni and Saad    |
!     -------------------------------------------

   USE GALAHAD_KINDS_precision
   USE GALAHAD_USEMIQR_precision

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_MIQR( input )

!  Close the data input file

   CLOSE( input  )
   STOP

!  End of RUNMIQR_SIF_precision

   END PROGRAM RUNMIQR_SIF_precision
