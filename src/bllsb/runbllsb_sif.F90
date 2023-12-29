! THIS VERSION: GALAHAD 4.3 - 2023-12-28 AT 11:00 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N B L L S B _ S I F  *-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 4.3, December 28th, 2023

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNBLLSB_SIF_precision

!     --------------------------------------------------
!    | Main program for the SIF/CUTEst interface to     |
!    | BLLSB, an interior-point crossover algorithm for |
!    | constrained linear least-squares optimization    |
!     --------------------------------------------------

   USE GALAHAD_KINDS_precision
   USE GALAHAD_USEBLLSB_precision

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEst interface

   CALL USE_BLLSB( input, close_input = .TRUE. )
   STOP

!  End of RUNBLLSB_SIF_precision

   END PROGRAM RUNBLLSB_SIF_precision
