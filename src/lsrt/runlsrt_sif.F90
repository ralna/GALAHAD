! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N L S R T _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 3.0. Feb 22nd 2017

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNLSRT_SIF_precision

!     -------------------------------------------------------
!    | Main program for the SIF/CUTEst interface to LSRT, an |
!    | iterative method for regularized linear least-squares |
!     -------------------------------------------------------

   USE GALAHAD_KINDS_precision
   USE GALAHAD_USELSRT_precision

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_LSRT( input )

!  Close the data input file

   CLOSE( input  )
   STOP

!  End of RUNLSRT_SIF_precision

   END PROGRAM RUNLSRT_SIF_precision
