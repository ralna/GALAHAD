! THIS VERSION: GALAHAD 2.5 - 10/04/2013 AT 15:30 GMT.

!-*-*-*-*-*-*-  G A L A H A D   R U N S H A _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  April 10th 2013

   PROGRAM RUNSHA_SIF
   USE GALAHAD_USESHA_double

!  Main program for the SIF interface to SHA, a method for estimating
!  sparse Hessian matrices

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

   CALL USE_SHA( insif )

!  Close the data input file 

   CLOSE( insif )
   STOP

!  End of RUNSHA_SIF

   END PROGRAM RUNSHA_SIF
