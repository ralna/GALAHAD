! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N S L L S _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould

!  History -
!   originally released with GALAHAD Version 4.1. March 26th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNBLLS_SIF_precision

!     --------------------------------------------------------
!    | Main program for the SIF/CUTEst interface to BLLS,     |
!    | a preconditiond projected conjugate-gradient algorithm |
!    | simplex-constrained linear least-squares minimization  |
!     --------------------------------------------------------

   USE GALAHAD_KINDS_precision
   USE GALAHAD_USEBLLS_precision

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEst interface

   CALL USE_BLLS( input )

!  Close the data input file

   CLOSE( input  )
   STOP

!  End of RUNBLLS_SIF_precision

   END PROGRAM RUNBLLS_SIF_precision
