! THIS VERSION: GALAHAD 3.3 - 25/05/2021 AT 08:30 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N S I L S _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 3.3. May 25th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNSILS_SIF

!    ------------------------------------------------------------
!    | Main program for the SIF/CUTEr interface to SILS, a      |
!    | method for solving symmetric systems of linear equations |
!    ------------------------------------------------------------

   USE GALAHAD_USESILS_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_SILS( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNSILS_SIF

   END PROGRAM RUNSILS_SIF
