! THIS VERSION: GALAHAD 2.2 - 15/05/2008 AT 14:30 GMT.

!-*-*-*-*-*-*-  G A L A H A D   R U N T R U _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  May 15th 2008

   PROGRAM RUNTRU_SIF
   USE GALAHAD_USETRU_double

!  Main program for the SIF interface to TRU, a trust-region algorithm for
!  unconstrained optimization

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

   CALL USE_TRU( insif )

!  Close the data input file 

   CLOSE( insif )
   STOP

!  End of RUNTRU_SIF

   END PROGRAM RUNTRU_SIF
