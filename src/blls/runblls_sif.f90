! THIS VERSION: GALAHAD 3.3 - 30/10/2019 AT 13:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N B L L S _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 3.3. October 30th 2019

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNBLLS_SIF

!     --------------------------------------------------------
!    | Main program for the SIF/CUTEr interface to BLLS,      |
!    | a preconditiond projected conjugate-gradient algorithm |
!    | bound-constrained linear least-squares minimization    |
!     --------------------------------------------------------

   USE GALAHAD_USEBLLS_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_BLLS( input )

!  Close the data input file

   CLOSE( input  )
   STOP

!  End of RUNBLLS_SIF

   END PROGRAM RUNBLLS_SIF
