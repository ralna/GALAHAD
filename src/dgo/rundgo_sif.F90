! THIS VERSION: GALAHAD 3.3 - 03/07/2021 AT 15:15 GMT.

!-*-*-*-*-*-*-  G A L A H A D   R U N D G O _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  July 3rd 2021

   PROGRAM RUNDGO_SIF
   USE GALAHAD_USEDGO_double

!  Main program for the SIF interface to DGO, an algorithm for
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

   CALL USE_DGO( insif )

!  Close the data input file

   CLOSE( insif )
   STOP

!  End of RUNDGO_SIF

   END PROGRAM RUNDGO_SIF
