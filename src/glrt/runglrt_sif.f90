! THIS VERSION: GALAHAD 3.0 - 27/10/2016 AT 11:35 GMT.

!-*-*-*-*-*-*-  G A L A H A D   R U N G L R T _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  October 24th 2016

   PROGRAM RUNGLRT_SIF
   USE GALAHAD_USEGLRT_double

!  Main program for the SIF interface to GLRT, a solver for the regularised
!  quadratic mninimization subproblem

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

   CALL USE_GLRT( insif )

!  Close the data input file

   CLOSE( insif )
   STOP

!  End of RUNGLRT_SIF

   END PROGRAM RUNGLRT_SIF
