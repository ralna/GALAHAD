! THIS VERSION: GALAHAD 3.0 - 22/02/2017 AT 10:50 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N L 2 R T _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 3.0. Feb 22nd 2017

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNL2RT_SIF

!     -----------------------------------------------------------------
!    | Main program for the SIF/CUTEst interface to L2RT, an iterative |
!    | method for regularized linear least-two-norm problems           |
!     -----------------------------------------------------------------

   USE GALAHAD_USEL2RT_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_L2RT( input )

!  Close the data input file

   CLOSE( input  )
   STOP

!  End of RUNL2RT_SIF

   END PROGRAM RUNL2RT_SIF
