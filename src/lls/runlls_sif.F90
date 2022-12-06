! THIS VERSION: GALAHAD 2.1 - 20/10/2007 AT 17:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N L L S _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Nick Gould and Dominique Orban, for GALAHAD productions
!  Copyright reserved
!  October 20th 2007

   PROGRAM RUNLLS_SIF
   USE GALAHAD_USELLS_double

!  Main program for the SIF/CUTEr interface to LLS, a conjugate-gradient 
!  algorithm for solving linear least-squares problems

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_LLS( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNLLS_SIF

   END PROGRAM RUNLLS_SIF
