! THIS VERSION: GALAHAD 2.4 - 19/01/2011 AT 08:30 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N S L S _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.4. January 19th 2011

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNSLS_SIF

!    ------------------------------------------------------------
!    | Main program for the SIF/CUTEr interface to SLS, a       |
!    | method for solving symmetric systems of linear equations |
!    ------------------------------------------------------------

   USE GALAHAD_USESLS_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_SLS( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNSLS_SIF

   END PROGRAM RUNSLS_SIF
