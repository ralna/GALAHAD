! THIS VERSION: GALAHAD 3.0 - 24/10/2017 AT 15:00 GMT.

!-*-*-*-*-*-*-  G A L A H A D   R U N P R E S O L V E _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  March 14th 2003

   PROGRAM RUNPRESOLVE_SIF
   USE GALAHAD_USEPRESOLVE_double

!  Main program for the SIF/CUTEr interface to PRESOLVE, a preprocessing
!  algorithm for quadratic programs

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_PRESOLVE( input )

!  Close the data input file

   CLOSE( input  )
   STOP

!  End of RUNPRESOLVE_SIF

   END PROGRAM RUNPRESOLVE_SIF
