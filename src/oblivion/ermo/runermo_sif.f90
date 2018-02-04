! THIS VERSION: GALAHAD 2.4 - 17/03/2009 AT 16:00 GMT.

!-*-*-*-*-*-*-  G A L A H A D   R U N E R M O _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  March 17th 2009

   PROGRAM RUNERMO_SIF
   USE GALAHAD_USEERMO_double

!  Main program for the SIF interface to ERMO, an enriched recursive multilevel
!  optimization algorithm for for unconstrained problems

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

   CALL USE_ERMO( insif )

!  Close the data input file 

   CLOSE( insif )
   STOP

!  End of RUNERMO_SIF

   END PROGRAM RUNERMO_SIF
