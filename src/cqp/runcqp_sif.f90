! THIS VERSION: GALAHAD 2.8 - 24/08/2016 AT 12:40 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N C Q P _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.4. January 1st 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNCQP_SIF

!    -----------------------------------------------
!    | Main program for the SIF/CUTEr interface to |
!    | CQP, an interior-point algorithm for convex |
!    | quadratic & least-distance programming      |
!    -----------------------------------------------

   USE GALAHAD_USECQP_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEst interface

   CALL USE_CQP( input, close_input = .TRUE. )
   STOP

!  End of RUNCQP_SIF

   END PROGRAM RUNCQP_SIF
