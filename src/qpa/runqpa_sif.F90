! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D   R U N Q P A B _ S I F  -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released pre GALAHAD Version 1.0. March 14th 2003
!   update released with GALAHAD Version 2.0. August 11th 2005

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   PROGRAM RUNQPA_SIF_precision

!    -----------------------------------------------------
!    | Main program for the SIF/CUTEr interface to QPA,  |
!    | a working-set algorithm for quadratic programming |
!    -----------------------------------------------------

   USE GALAHAD_KINDS_precision
   USE GALAHAD_USEQPA_precision
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

   CALL USE_QPA( input )

!  Close the data input file

   CLOSE( input  )

!  Unload the shared library

#ifdef CUTEST_SHARED
   CALL GALAHAD_unload_routines()
#endif

   STOP

!  End of RUNQPA_SIF_precision

   END PROGRAM RUNQPA_SIF_precision
