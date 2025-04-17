! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-  G A L A H A D   R U N T R B _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  June 25th 2012

   PROGRAM RUNTRB_SIF_precision
   USE GALAHAD_KINDS_precision
   USE GALAHAD_USETRB_precision
#ifdef CUTEST_SHARED
   USE CUTEST_TRAMPOLINE_precision
   USE iso_c_binding, ONLY: c_null_char
#endif

!  Main program for the SIF interface to TRB, a trust-region algorithm for
!  bound-constrained optimization

!  Problem insif characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: errout = 6
   INTEGER ( KIND = ip_ ), PARAMETER :: insif = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'
   INTEGER ( KIND = ip_ ) :: iostat

!  Load the shared library

#ifdef CUTEST_SHARED
   CHARACTER ( LEN = 256 ) :: libsif_path
   INTEGER :: arg_len

   CALL GET_COMMAND_ARGUMENT(1, libsif_path, LENGTH = arg_len)
   IF (arg_len <= 0) THEN
      WRITE(*,*) 'ERROR: please provide the path to the shared library'
      STOP 1
   END IF

   CALL CUTEST_LOAD_ROUTINES(TRIM(libsif_path) // c_null_char)
#endif

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

   CALL USE_TRB( insif )

!  Close the data input file

   CLOSE( insif )

!  Unload the shared library

#ifdef CUTEST_SHARED
   CALL CUTEST_UNLOAD_ROUTINES()
#endif

   STOP

!  End of RUNTRB_SIF_precision

   END PROGRAM RUNTRB_SIF_precision
