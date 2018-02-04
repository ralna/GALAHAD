! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N L C F _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.0.July 20th 2006

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNLCF_SIF

!    ----------------------------------------------------
!    | Main program for the SIF/CUTEr interface to LCF, |
!    | an alternating projection algorithm for finding  |
!    | a feasible point within a polyhedron             |
!    ---------------------------------------------------

   USE GALAHAD_USELCF_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_LCF( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNLCF_SIF

   END PROGRAM RUNLCF_SIF
