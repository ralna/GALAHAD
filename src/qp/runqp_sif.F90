! THIS VERSION: GALAHAD 2.4 - 05/01/2011 AT 07:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N Q P _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.4. January 5th 2011

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNQP_SIF

!    --------------------------------------------------------
!    | Main program for the SIF/CUTEr interface to QP,      |
!    | a generic QP method that allows uniform access       |
!    | to other GALAHAD QP solvers                          |
!    --------------------------------------------------------

   USE GALAHAD_USEQP_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_QP( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNQP_SIF

   END PROGRAM RUNQP_SIF
