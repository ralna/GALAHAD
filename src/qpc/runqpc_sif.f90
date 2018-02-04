! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N Q P C _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.0. August 11th 2005

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNQPC_SIF

!    -----------------------------------------------------
!    | Main program for the SIF/CUTEr interface to QPC,  |
!    | an interior-point/working-set crossover algorithm |
!    | for quadratic programming                         |
!    -----------------------------------------------------

   USE GALAHAD_USEQPC_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_QPC( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNQPC_SIF

   END PROGRAM RUNQPC_SIF
