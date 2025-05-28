! THIS VERSION: GALAHAD 5.2 - 2025-03-11 AT 08:45 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-  G A L A H A D   R U N N O D E N D _ S I F  *-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 5.2. March 11th 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNNODEND_SIF_precision

!     -----------------------------------------------------------
!    | Main program for CUTEst/AMPL interface to METIS_nodend, a  |
!    | method for the nested-ordering of symmetric sparse matices |
!     -----------------------------------------------------------

   USE GALAHAD_KINDS_precision
   USE GALAHAD_USENODEND_precision
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

!  Call the CUTEr interface

   CALL USE_NODEND( input )

!  Close the data input file

   CLOSE( input  )

!  Unload the shared library

#ifdef CUTEST_SHARED
   CALL GALAHAD_unload_routines()
#endif

   STOP

!  End of RUNNODEND_SIF_precision

   END PROGRAM RUNNODEND_SIF_precision
