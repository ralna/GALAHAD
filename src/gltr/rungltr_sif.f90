! THIS VERSION: GALAHAD 3.0 - 24/10/2016 AT 14:00 GMT.

!-*-*-*-*-*-*-  G A L A H A D   R U N G L T R _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  October 24th 2016

   PROGRAM RUNGLTR_SIF
   USE GALAHAD_USEGLTR_double

!  Main program for the SIF interface to GLTR, a solver for the trust-region
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

   CALL USE_GLTR( insif )

!  Close the data input file

   CLOSE( insif )
   STOP

!  End of RUNGLTR_SIF

   END PROGRAM RUNGLTR_SIF
