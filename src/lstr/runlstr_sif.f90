! THIS VERSION: GALAHAD 2.6 - 27/05/2014 AT 14:30 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N L S T R _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.6. May 27th 2014

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNLSTR_SIF

!     -----------------------------------------------
!    | Main program for the SIF/CUTEst interface     |
!    | to LSTR, an iterative method for trust-region |
!    | regularized linear least-squares              |
!     -----------------------------------------------

   USE GALAHAD_USELSTR_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_LSTR( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNLSTR_SIF

   END PROGRAM RUNLSTR_SIF
