! THIS VERSION: GALAHAD 2.4 - 27/02/2009 AT 14:00 GMT.

!-*-*-*-*-*-*-  G A L A H A D   R U N I S M _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  February 27th 2009

   PROGRAM RUNISM_SIF
   USE GALAHAD_USEISM_double

!  Main program for the SIF interface to ISM, an iterated-subspace minimization
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

   CALL USE_ISM( insif )

!  Close the data input file 

   CLOSE( insif )
   STOP

!  End of RUNISM_SIF

   END PROGRAM RUNISM_SIF
