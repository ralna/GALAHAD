! THIS VERSION: GALAHAD 2.2 - 07/02/2008 AT 17:00 GMT.

!-*-*-*-*-*-*-  G A L A H A D   R U N B A R C _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  February 7th 2008

   PROGRAM RUNBARC_SIF
   USE GALAHAD_USEBARC_double

!  Main program for the SIF interface to BARC, an adaptive cubic overestimation
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

   CALL USE_BARC( insif )

!  Close the data input file 

   CLOSE( insif )
   STOP

!  End of RUNBARC_SIF

   END PROGRAM RUNBARC_SIF
