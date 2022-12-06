! THIS VERSION: GALAHAD 2.3 - 15/04/2009 AT 13:00 GMT.

!-*-*-*-*-*-*-  G A L A H A D   R U N D E M O _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  April 15th 2009

   PROGRAM RUNDEMO_SIF
   USE GALAHAD_USEDEMO_double

!  Main program for the SIF interface to DEMO, a solver for the demost-region 
!  subproblem

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

   CALL USE_DEMO( insif )

!  Close the data input file 

   CLOSE( insif )
   STOP

!  End of RUNDEMO_SIF

   END PROGRAM RUNDEMO_SIF
