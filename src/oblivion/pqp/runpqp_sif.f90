! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D   R U N P Q P _ S I F  *-*-*-*-*-*-

!  Nick Gould and Dominique Orban, for GALAHAD productions
!  Copyright reserved
!  September 14th 2004

   PROGRAM RUNPQP_SIF
   USE GALAHAD_USEPQP_double

!  Main program for the SIF/CUTEr interface to QPA, a working-set 
!  algorithm for solving quadratic programs

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_PQP( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNPQP_SIF

   END PROGRAM RUNPQP_SIF
