! THIS VERSION: GALAHAD 3.1 - 07/10/2018 AT 12:05 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N L P A _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.1. October 7th 2018

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNLPA_SIF

!    ---------------------------------------------------------
!    | Main program for the SIF/CUTEr interface to LPA, an   |
!    | active-set (simplex) algorithm for linear programming |
!    ---------------------------------------------------------

   USE GALAHAD_USELPA_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEst interface

   CALL USE_LPA( input, close_input = .TRUE. )
   STOP

!  End of RUNLPA_SIF

   END PROGRAM RUNLPA_SIF
