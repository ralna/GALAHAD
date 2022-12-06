! THIS VERSION: GALAHAD 4.1 - 2022-11-25 AT 16:30 GMT.

!-*-*-*-  G A L A H A D  -  D U M M Y   M P I   S U B R O U T I N E  -*-*-*-*-

   SUBROUTINE MPI_INIT( ierr )
   INTEGER :: ierr
   ierr = - 1 ! error code
   END SUBROUTINE MPI_INIT

   SUBROUTINE MPI_INITIALIZED( flag, ierr )
   LOGICAL :: flag
   INTEGER :: ierr
   flag = .FALSE.
   ierr = - 1 ! error code
   END SUBROUTINE MPI_INITIALIZED

   SUBROUTINE MPI_FINALIZE( ierr )
   INTEGER :: ierr
   ierr = - 1 ! error code
   END SUBROUTINE MPI_FINALIZE
