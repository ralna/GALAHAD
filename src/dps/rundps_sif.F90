! THIS VERSION: GALAHAD 3.0 - 15/03/2018 AT 15:40 GMT.

!-*-*-*-*-*-*-  G A L A H A D   R U N D P S _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  March 15th 2018

   PROGRAM RUNDPS_SIF
   USE GALAHAD_USEDPS_double

!  Main program for the SIF interface to DPS, a solver for trust-region and
!  regularized quadratic subproblems in a variety of diagonalizing norms

!  Problem insif characteristics

   INTEGER, PARAMETER :: errout = 6
   INTEGER, PARAMETER :: insif = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'
   INTEGER :: iostat

!  Open the data input file

   OPEN( insif, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD',             &
         IOSTAT = iostat )
   IF ( iostat > 0 ) THEN
     WRITE( errout,                                                            &
       "( ' ERROR: could not open file OUTSDIF.d on unit ', I2 )" ) insif
     STOP
   END IF
   REWIND insif

!  Call the SIF interface

   CALL USE_DPS( insif )

!  Close the data input file

   CLOSE( insif )
   STOP

!  End of RUNDPS_SIF

   END PROGRAM RUNDPS_SIF
