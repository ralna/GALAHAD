! THIS VERSION: GALAHAD 2.8 - 20/06/2016 AT 15:15 GMT.

!-*-*-*-*-*-*-  G A L A H A D   R U N B G O _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  June 20th 2016

   PROGRAM RUNBGO_SIF
   USE GALAHAD_USEBGO_double

!  Main program for the SIF interface to BGO, an algorithm for
!  bound-constrained gobal optimization

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

   CALL USE_BGO( insif )

!  Close the data input file

   CLOSE( insif )
   STOP

!  End of RUNBGO_SIF

   END PROGRAM RUNBGO_SIF
