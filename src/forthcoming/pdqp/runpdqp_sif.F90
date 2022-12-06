! THIS VERSION: GALAHAD 3.3 - 14/04/2021 AT 08:30 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N P D Q P _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released GALAHAD Version 2.4. August 22nd 2009 as QPE
!   renamed as PDQP, GALAHAD Version 3.3, April 14th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNPDQP_SIF

!    -----------------------------------------------------
!    | Main program for the SIF/CUTEr interface to QPE,  |
!    | a primal-dual active-set algorithm for M-convex   |
!    | quadratic programming                             |
!    -----------------------------------------------------

   USE GALAHAD_USEPDQP_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_PDQP( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNQPE_SIF

   END PROGRAM RUNPDQP_SIF
