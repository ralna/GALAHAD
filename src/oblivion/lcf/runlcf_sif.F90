! THIS VERSION: GALAHAD 4.1 - 2022-12-29 AT 14:20 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N L C F _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.0.July 20th 2006

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNLCF_SIF

!    ----------------------------------------------------
!    | Main program for the SIF/CUTEr interface to LCF, |
!    | an alternating projection algorithm for finding  |
!    | a feasible point within a polyhedron             |
!    ---------------------------------------------------

   USE GALAHAD_PRECISION
   USE GALAHAD_USELCF_precision

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_LCF( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNLCF_SIF

   END PROGRAM RUNLCF_SIF
