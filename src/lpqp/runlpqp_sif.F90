! THIS VERSION: GALAHAD 3.3 - 05/05/2021 AT 08:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N L P Q P _ S I F  *-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 3.3. May 5th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNLPQP_SIF

!    ----------------------------------------------------------------
!    | Main program for the SIF/CUTEr interface to  LPQP, a program |
!    | to assemble an l_p QP from an input quadratic program        |
!    ----------------------------------------------------------------

   USE GALAHAD_USELPQP_double

!  Problem input characteristics

   INTEGER, PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEst interface

   CALL USE_LPQP( input, close_input = .TRUE. )
   STOP

!  End of RUNLPQP_SIF

   END PROGRAM RUNLPQP_SIF
