! THIS VERSION: GALAHAD 2.8 - 24/08/2016 AT 12:40 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N S B L S _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.4. January 18th 2011

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNSBLS_SIF

!    --------------------------------------------------------
!    | Main program for the SIF/CUTEr interface to SBLS, a  |
!    | method for solving block systems of linear equations |
!    --------------------------------------------------------

   USE GALAHAD_USESBLS_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d       '

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEst interface

   CALL USE_SBLS( input, close_input = .TRUE. )
   STOP

!  End of RUNSBLS_SIF

   END PROGRAM RUNSBLS_SIF
