! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-  G A L A H A D   R U N F I L T R A N E _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  June 9th 2003

   PROGRAM RUNFILTRANE_SIF
   USE GALAHAD_USEFILTRANE_double

!  Main program for the SIF/CUTEr interface to FILTRANE, a filter
!  trust-region method for feasibility problems.

!  Problem input characteristics

   INTEGER, PARAMETER :: errout = 6
   INTEGER, PARAMETER :: insif  = 56      ! OUTSDIF.d device number
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'
   INTEGER :: iostat

!  Open the data input file

   OPEN( insif, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD',              &
         IOSTAT = iostat )

   IF ( iostat > 0 ) THEN
     WRITE( errout,                                                             &
       "( ' ERROR: could not open file OUTSDIF.d on unit ', I2 )" ) insif
     STOP
   END IF

   REWIND insif

!  Call the CUTEr interface

   CALL USE_FILTRANE( insif )

!  Close the data input file 

   CLOSE( insif  )
   STOP

!  End of RUNFILTRANE_SIF

   END PROGRAM RUNFILTRANE_SIF
