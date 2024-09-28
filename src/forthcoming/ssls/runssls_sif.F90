! THIS VERSION: GALAHAD 5.1 - 2024-09-10 AT 14:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N S B L S _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 5.1. September 10th 2024

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNSSLS_SIF_precision

!    --------------------------------------------------------
!    | Main program for the SIF/CUTEr interface to SSLS, a  |
!    | method for solving block systems of linear equations |
!    --------------------------------------------------------

   USE GALAHAD_KINDS_precision
   USE GALAHAD_USESSLS_precision

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d       '

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEst interface

   CALL USE_SSLS( input, close_input = .TRUE. )
   STOP

!  End of RUNSSLS_SIF_precision

   END PROGRAM RUNSSLS_SIF_precision
