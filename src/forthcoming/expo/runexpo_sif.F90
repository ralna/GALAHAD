! THIS VERSION: GALAHAD 5.3 - 2024-06-15 AT 11:00 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-  G A L A H A D   R U N E X P O _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould and Alexis Montoison, for GALAHAD productions
!  Copyright reserved
!  June 15th 2025

   PROGRAM RUNEXPO_SIF_precision
   USE GALAHAD_KINDS_precision
   USE GALAHAD_USEEXPO_precision
   USE GALAHAD_CUTEST_precision
   USE ISO_C_BINDING, ONLY : C_NULL_CHAR

!  Main program for the SIF interface to EXPO, an exponential penalty function
!  algorithm for constrained optimization

!  Problem insif characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: errout = 6
   INTEGER ( KIND = ip_ ), PARAMETER :: insif = 55
   CHARACTER ( LEN = 256 ) :: prbdat = 'OUTSDIF.d'
   INTEGER ( KIND = ip_ ) :: iostat

!  Load the shared library

#ifdef CUTEST_SHARED
   CHARACTER ( LEN = 256 ) :: libsif_path
   CHARACTER ( LEN = 256 ) :: outsdif_path
   INTEGER :: arg_len
   INTEGER :: arg_len2

   CALL GET_COMMAND_ARGUMENT(1, libsif_path, LENGTH = arg_len)
   IF (arg_len <= 0) THEN
      WRITE(*,*) 'ERROR: please provide the path to the shared library'
      STOP 1
   END IF
   CALL GET_COMMAND_ARGUMENT(2, outsdif_path, LENGTH = arg_len2)
   IF (arg_len2 > 0) THEN
      prbdat = TRIM(outsdif_path)
      WRITE(*,*) 'Using OUTSDIF file:', prbdat
   END IF

   CALL GALAHAD_load_routines(TRIM(libsif_path) // C_NULL_CHAR)
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

   CALL USE_EXPO( insif )

!  Close the data input file

   CLOSE( insif )

!  Unload the shared library

#ifdef CUTEST_SHARED
   CALL GALAHAD_unload_routines()
#endif

   STOP

!  End of RUNEXPO_SIF_precision

   END PROGRAM RUNEXPO_SIF_precision
