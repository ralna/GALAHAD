! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N Q P C _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.0. November 1st 2005

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNWCP_SIF

!    ----------------------------------------------------
!    | Main program for the SIF/CUTEr interface to WCP, |
!    | an interior-point algorithm for finding a        |
!    | well-centered point within a polyhedron          |
!    ---------------------------------------------------

   USE GALAHAD_USEWCP_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEr interface

   CALL USE_WCP( input )

!  Close the data input file 

   CLOSE( input  )
   STOP

!  End of RUNWCP_SIF

   END PROGRAM RUNWCP_SIF
