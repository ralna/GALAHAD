! THIS VERSION: GALAHAD 4.1 - 2022-12-30 AT 09:40 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N B Q P _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.4. January 1st 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNBQP_SIF

!    ----------------------------------------------------------
!    | Main program for the SIF/CUTEr interface to BQP,       |
!    | a preconditiond projected conjugate-gradient algorithm |
!    | for bound-constrained convex quadratic programming     |
!    ----------------------------------------------------------

   USE GALAHAD_KINDS
   USE GALAHAD_USEBQP_precision

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_BQP( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNBQP_SIF

   END PROGRAM RUNBQP_SIF
