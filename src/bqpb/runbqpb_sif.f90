! THIS VERSION: GALAHAD 2.4 - 18/01/2010 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N B Q P B _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.4. January 18th 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNBQPB_SIF

!    -------------------------------------------------------------
!    | Main program for the SIF/CUTEr interface to BQPB, a       |
!    | preconditiond interior-point conjugate-gradient algorithm |
!    | for bound-constrained convex quadratic programming        |
!    -------------------------------------------------------------

   USE GALAHAD_USEBQPB_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_BQPB( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNBQPB_SIF

   END PROGRAM RUNBQPB_SIF
