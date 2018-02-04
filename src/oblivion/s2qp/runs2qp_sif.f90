  ! THIS VERSION: GALAHAD 2.1 - 22/12/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N S 2 Q P _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould and Daniel Robinson, for GALAHAD productions
!  Copyright reserved
!  December 22nd 2007

   PROGRAM RUNS2QP_SIF
   USE GALAHAD_USES2QP_double

!  Main program for the SIF interface to S2QP, a trust-region SQP
!  method algorithm for nonlinear programming, in which descent is
!  imposed as an additional explicit constraint in the subproblem.

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

   CALL USE_S2QP( insif )

!  Close the data input file 

   CLOSE( insif  )
   STOP

!  End of RUNS2QP_SIF

   END PROGRAM RUNS2QP_SIF
