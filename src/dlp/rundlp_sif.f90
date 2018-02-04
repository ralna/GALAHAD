! THIS VERSION: GALAHAD 2.6 - 30/01/2015 AT 15:05 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N D L P _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.6. January 30th 2015

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNDLP_SIF

!     ------------------------------------------
!    | Main program for the SIF/CUTEr interface |
!    | to DLP, dual gradient-projection         |
!    | algorithm for linear programming         |
!     ------------------------------------------

   USE GALAHAD_USEDLP_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_DLP( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNDLP_SIF

   END PROGRAM RUNDLP_SIF
