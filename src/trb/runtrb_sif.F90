! THIS VERSION: GALAHAD 2.5 - 25/06/2011 AT 14:30 GMT.

!-*-*-*-*-*-*-  G A L A H A D   R U N T R B _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  June 25th 2012

   PROGRAM RUNTRB_SIF
   USE GALAHAD_USETRB_double

!  Main program for the SIF interface to TRB, a trust-region algorithm for
!  bound-constrained optimization

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

   CALL USE_TRB( insif )

!  Close the data input file 

   CLOSE( insif )
   STOP

!  End of RUNTRB_SIF

   END PROGRAM RUNTRB_SIF
