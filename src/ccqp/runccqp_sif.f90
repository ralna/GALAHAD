! THIS VERSION: GALAHAD 2.7 - 17/07/2015 AT 13:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N C C Q P _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.7. July 17th 2015

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNCCQP_SIF

!     -------------------------------------------------------------------
!    | Main program for the SIF/CUTEr interface to CCQP, an              |
!    | infeasible primal-dual interior-point to dual gradient-projection |
!    | crossover method for convex quadratic programming                 |
!     -------------------------------------------------------------------

   USE GALAHAD_USECCQP_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_CCQP( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNCCQP_SIF

   END PROGRAM RUNCCQP_SIF
