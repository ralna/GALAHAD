! THIS VERSION: GALAHAD 5.2 - 2025-03-11 AT 08:45 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-  G A L A H A D   R U N N O D E N D _ S I F  *-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 5.2. March 11th 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNNODEND_SIF_precision

!     -----------------------------------------------------------
!    | Main program for CUTEst/AMPL interface to METIS_nodend, a  |
!    | method for the nested-ordering of symmetric sparse matices |
!     -----------------------------------------------------------

   USE GALAHAD_KINDS_precision
   USE GALAHAD_USENODEND_precision

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_NODEND( input )

!  Close the data input file

   CLOSE( input  )
   STOP

!  End of RUNNODEND_SIF_precision

   END PROGRAM RUNNODEND_SIF_precision
