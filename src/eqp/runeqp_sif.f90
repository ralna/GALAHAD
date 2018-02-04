! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N E Q P _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Nick Gould and Dominique Orban, for GALAHAD productions
!  Copyright reserved
!  March 25th 2004

   PROGRAM RUNEQP_SIF
   USE GALAHAD_USEEQP_double

!  Main program for the SIF/CUTEr interface to EQP, a projected
!  conjugate-gradient algorithm for solving equality-constrained 
!  quadratic programs

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_EQP( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNEQP_SIF

   END PROGRAM RUNEQP_SIF
