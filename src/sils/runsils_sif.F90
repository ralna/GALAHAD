! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N S I L S _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 3.3. May 25th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNSILS_SIF_precision

!    ------------------------------------------------------------
!    | Main program for the SIF/CUTEr interface to SILS, a      |
!    | method for solving symmetric systems of linear equations |
!    ------------------------------------------------------------

   USE GALAHAD_KINDS_precision
   USE GALAHAD_USESILS_precision

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_SILS( input )

!  Close the data input file

   CLOSE( input  )
   STOP

!  End of RUNSILS_SIF_precision

   END PROGRAM RUNSILS_SIF_precision
