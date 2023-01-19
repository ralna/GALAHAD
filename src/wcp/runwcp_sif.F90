! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N Q P C _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.0. November 1st 2005

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNWCP_SIF_precision

!    ----------------------------------------------------
!    | Main program for the SIF/CUTEr interface to WCP, |
!    | an interior-point algorithm for finding a        |
!    | well-centered point within a polyhedron          |
!    ---------------------------------------------------

   USE GALAHAD_USEWCP_precision
   USE GALAHAD_KINDS_precision

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_WCP( input )

!  Close the data input file

   CLOSE( input  )
   STOP

!  End of RUNWCP_SIF_precision

   END PROGRAM RUNWCP_SIF_precision
