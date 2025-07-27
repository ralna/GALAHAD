! THIS VERSION: GALAHAD 5.1 - 2024-09-10 AT 14:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N S B L S _ S I F  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 5.1. September 10th 2024

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNSSLS_SIF_precision

!    --------------------------------------------------------
!    | Main program for the SIF/CUTEr interface to SSLS, a  |
!    | method for solving block systems of linear equations |
!    --------------------------------------------------------

   USE GALAHAD_KINDS_precision
   USE GALAHAD_USESSLS_precision
   USE GALAHAD_CUTEST_precision
   USE ISO_C_BINDING, ONLY : C_NULL_CHAR

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 256 ) :: prbdat = 'OUTSDIF.d'

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

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the CUTEst interface

   CALL USE_SSLS( input, close_input = .TRUE. )

!  Unload the shared library

#ifdef CUTEST_SHARED
   CALL GALAHAD_unload_routines()
#endif

   STOP

!  End of RUNSSLS_SIF_precision

   END PROGRAM RUNSSLS_SIF_precision
