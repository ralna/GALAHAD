! THIS VERSION: GALAHAD 3.0 - 29/06/2017 AT 12:30 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N L 1 Q P _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 3.0. June 29th 2017

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNL1QP_SIF

!     -------------------------------------------------------------------
!    | Main program for the SIF/CUTEr interface to L1QP, an              |
!    | infeasible primal-dual interior-point to dual gradient-projection |
!    | crossover method for convex quadratic programming                 |
!     -------------------------------------------------------------------

   USE GALAHAD_USEL1QP_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_L1QP( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNL1QP_SIF

   END PROGRAM RUNL1QP_SIF
