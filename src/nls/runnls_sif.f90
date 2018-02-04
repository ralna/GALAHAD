! THIS VERSION: GALAHAD 3.0 - 25/11/2016 AT 09:15 GMT

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N N L S _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  October 27th 2015

   PROGRAM RUNNLS_SIF
   USE GALAHAD_USENLS_double

!  Main program for the SIF interface to NLS, a regularization algorithm for
!  nonlinear least-squares optimization

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

   CALL USE_NLS( insif )

!  Close the data input file

   CLOSE( insif )
   STOP

!  End of RUNNLS_SIF

   END PROGRAM RUNNLS_SIF
