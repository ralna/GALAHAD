! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N S Q P _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  March 8th 2006

   PROGRAM RUNSQP_SIF
   USE GALAHAD_USESQP_double

!  Main program for the SIF interface to SQP, a primative SQP
!  algorithm for nonlinear programming

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

!  CALL USE_SQP( insif )
   CALL USE_SQP_DPR( insif )

!  Close the data input file 

   CLOSE( insif  )
   STOP

!  End of RUNSQP_SIF

   END PROGRAM RUNSQP_SIF
