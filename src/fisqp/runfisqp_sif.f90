! THIS VERSION: GALAHAD 2.6 - 24/11/2014 AT 10:30 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N F I S Q P _ S I F  *-*-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  November 24th 2014

   PROGRAM RUNFISQP_SIF
   USE GALAHAD_USEFISQP_double

!  Main program for the SIF interface to FiSQP, a filter SQP algorithm 
!  with unified step calculation for nonlinear programming

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

   CALL USE_FISQP( insif )

!  Close the data input file 

   CLOSE( insif  )
   STOP

!  End of RUNFISQP_SIF

   END PROGRAM RUNFISQP_SIF
