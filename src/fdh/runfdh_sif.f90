! THIS VERSION: GALAHAD 2.6 - 11/10/2013 AT 13:30 GMT.

!-*-*-*-*-*-*-  G A L A H A D   R U N F D H _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  October 11th 2013

   PROGRAM RUNFDH_SIF
   USE GALAHAD_USEFDH_double

!  Main program for the SIF interface to FDH, a method for estimating
!  sparse Hessian matrices by differences

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

   CALL USE_FDH( insif )

!  Close the data input file 

   CLOSE( insif )
   STOP

!  End of RUNFDH_SIF

   END PROGRAM RUNFDH_SIF
