! THIS VERSION: GALAHAD 2.2 - 27/10/2007 AT 16:30 GMT.

!-*-*-*-*-*-*-  G A L A H A D   R U N A R C _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  October 27th 2007

   PROGRAM RUNARC_SIF
   USE GALAHAD_USEARC_double

!  Main program for the SIF interface to ARC, an adaptive cubic overestimation
!  algorithm for unconstrained optimization

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

   CALL USE_ARC( insif )

!  Close the data input file 

   CLOSE( insif )
   STOP

!  End of RUNARC_SIF

   END PROGRAM RUNARC_SIF
