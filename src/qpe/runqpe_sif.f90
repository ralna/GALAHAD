! THIS VERSION: GALAHAD 2.4 - 22/08/2009 AT 17:30 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N Q P E _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released GALAHAD Version 2.4. August 22nd 2009

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNQPE_SIF

!    -----------------------------------------------------
!    | Main program for the SIF/CUTEr interface to QPE,  |
!    | a primal-dual active-set algorithm for convex     |
!    | quadratic programming                             |
!    -----------------------------------------------------

   USE GALAHAD_USEQPE_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_QPE( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNQPE_SIF

   END PROGRAM RUNQPE_SIF
