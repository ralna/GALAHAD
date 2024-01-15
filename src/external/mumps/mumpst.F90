! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
! This file is a modified version of dsimpletest.F from MUMPS 5.5.1,
! originally released on Tue Jul 12 13:17:24 UTC 2022

#include "galahad_modules.h"

   PROGRAM MUMPS_TEST
   USE GALAHAD_KINDS_precision
   USE GALAHAD_MUMPS_TYPES_precision
   IMPLICIT NONE
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 5
   INTEGER ( KIND = ip_ ), PARAMETER :: ne = 7
   INTEGER ( KIND = ip_ ), PARAMETER :: neu = 11
   TYPE ( MUMPS_STRUC ) mumps_par
   INTEGER ( KIND = ip_ ) :: ierr, j, nnz

!  start an mpi instance, and define a communicator for the package

   CALL MPI_INIT( ierr )
   IF ( ierr < 0 ) THEN
     WRITE( 6, "( ' No MPi available, stopping' )" )
     STOP
   END IF
   mumps_par%COMM = MPI_COMM_WORLD

   DO j = 0, 1  ! loop over unsymmetric and symmetric examples

!  initialize an instance of the package for L U or L D L^T factorization
!  (sym = 0, with working host)

      mumps_par%JOB = - 1
      mumps_par%SYM = j
      mumps_par%PAR = 1
      CALL MUMPS_precision( mumps_par )
      IF ( mumps_par%INFOG( 1 ) == - 999 ) THEN
        WRITE( 6, "( ' Mumps not provided, stopping' )" )
        EXIT
      ELSE IF ( mumps_par%INFOG( 1 ) < 0 ) THEN
        WRITE( 6, 10 ) 'Initialize', mumps_par%INFOG( 1 : 2 )
        EXIT
      END IF

!  define problem on the host (processor 0)

      IF ( mumps_par%MYID == 0 ) THEN
        mumps_par%N = n
        IF ( j == 0 ) THEN
          nnz = neu
        ELSE
          nnz = ne
        END IF
        mumps_par%NNZ = nnz
        ALLOCATE( mumps_par%IRN( nnz ), mumps_par%JCN( nnz ),                  &
                  mumps_par%A( nnz ), mumps_par%RHS( n ) )
        IF ( j == 0 ) THEN ! unsymmetric matrix
          mumps_par%IRN ( : nnz ) = (/ 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 5 /)
          mumps_par%JCN ( : nnz ) = (/ 1, 2, 1, 3, 5, 2, 3, 4, 3, 2, 5 /)
          mumps_par%A( : nnz ) = (/ 2.0_rp_, 3.0_rp_, 3.0_rp_, 4.0_rp_,        &
                                    6.0_rp_, 4.0_rp_, 1.0_rp_, 5.0_rp_,        &
                                    5.0_rp_, 6.0_rp_, 1.0_rp_ /)
        ELSE ! symmetric matrix
          mumps_par%IRN ( : nnz ) = (/ 1, 2, 3, 3, 4, 5, 5 /)
          mumps_par%JCN ( : nnz ) = (/ 1, 1, 2, 3, 3, 2, 5 /)
          mumps_par%A( : nnz ) = (/ 2.0_rp_, 3.0_rp_, 4.0_rp_, 1.0_rp_,        &
                                    5.0_rp_, 6.0_rp_, 1.0_rp_ /)
        END IF
        mumps_par%RHS( : n ) = (/ 8.0_rp_, 45.0_rp_, 31.0_rp_, 15.0_rp_,       &
                                  17.0_rp_ /)
      END IF

!  call package for analysis

      mumps_par%JOB = 1
      CALL MUMPS_precision( mumps_par )
      IF ( mumps_par%INFOG( 1 ) < 0 ) THEN
        WRITE( 6, 10 ) 'Analyse', mumps_par%INFOG( 1 : 2 )
        EXIT
      END IF

!  call package for factorization

      mumps_par%JOB = 2
      CALL MUMPS_precision( mumps_par )
      IF ( mumps_par%INFOG( 1 ) < 0 ) THEN
        WRITE( 6, 10 ) 'Factorize', mumps_par%INFOG( 1 : 2 )
        EXIT
      END IF

!  call package for solution

      mumps_par%JOB = 3
      CALL MUMPS_precision( mumps_par )
      IF ( mumps_par%INFOG( 1 ) < 0 ) THEN
        WRITE( 6, 10 ) 'Solve', mumps_par%INFOG( 1 : 2 )
        EXIT
      END IF

!  solution has been assembled on the host

      IF ( mumps_par%MYID == 0 )                                               &
        WRITE( 6, "( ' Solution is ', 5ES12.4 )" ) mumps_par%RHS( 1 : n )

!  deallocate user data

      IF ( mumps_par%MYID == 0 )                                               &
        DEALLOCATE( mumps_par%IRN, mumps_par%JCN, mumps_par%A, mumps_par%RHS )

!  destroy the instance (deallocate internal data structures)

      mumps_par%JOB = - 2
      CALL MUMPS_precision( mumps_par )
      IF ( mumps_par%INFOG( 1 ) < 0 ) THEN
        WRITE( 6, 10 ) 'Terminate', mumps_par%INFOG( 1 : 2 )
        EXIT
      END IF
   END DO

!  terminate the mpi instance

   CALL MPI_FINALIZE( ierr )
   STOP
10 FORMAT( 1X, A, ' error return: mumps_par%infog(1,2) = ', I0, ', ', I0 )
   END PROGRAM MUMPS_TEST
