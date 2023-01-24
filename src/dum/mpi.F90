! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-  G A L A H A D  -  D U M M Y   M P I   S U B R O U T I N E  -*-*-*-*-

   SUBROUTINE MPI_INIT( ierr )
   USE GALAHAD_KINDS_precision
   INTEGER ( KIND = ip_ ) :: ierr
   ierr = - 1 ! error code
   END SUBROUTINE MPI_INIT

   SUBROUTINE MPI_INITIALIZED( flag, ierr )
   USE GALAHAD_KINDS_precision
   LOGICAL :: flag
   INTEGER ( KIND = ip_ ) :: ierr
   flag = .FALSE.
   ierr = - 1 ! error code
   END SUBROUTINE MPI_INITIALIZED

   SUBROUTINE MPI_FINALIZE( ierr )
   USE GALAHAD_KINDS_precision
   INTEGER ( KIND = ip_ ) :: ierr
   ierr = - 1 ! error code
   END SUBROUTINE MPI_FINALIZE
