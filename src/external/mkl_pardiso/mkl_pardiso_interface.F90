! THIS VERSION: GALAHAD 4.1 - 2022-12-30 AT 09:40 GMT.

#include "galahad_modules.h"

!- G A L A H A D  -  M K L  P A R A D I S O  I N T E R F A C E   M O D U L E S -

MODULE MKL_PARDISO_PRIVATE
  USE GALAHAD_KINDS, ONLY: ip_, long_
  TYPE MKL_PARDISO_HANDLE
    INTEGER ( KIND = long_ ) DUMMY
  END TYPE
  INTEGER ( KIND = ip_ ), PARAMETER :: PARDISO_OOC_FILE_NAME = 1
END MODULE MKL_PARDISO_PRIVATE

MODULE MKL_PARDISO
  USE MKL_PARDISO_PRIVATE

  INTERFACE MKL_PARDISO_SOLVE
    SUBROUTINE PARDISO_S( PT, MAXFCT, MNUM, MTYPE, PHASE, N, A, IA, JA,        &
                          PERM, NRHS, IPARM, MSGLVL, B, X, ERROR )
      USE MKL_PARDISO_PRIVATE
      USE GALAHAD_KINDS, ONLY: sp_, ip_
      TYPE( MKL_PARDISO_HANDLE ), INTENT( INOUT ) :: PT( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MAXFCT
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MNUM
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MTYPE
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: PHASE
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: N
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: IA( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: JA( * )
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: PERM( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: NRHS
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: IPARM( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MSGLVL
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: ERROR
      REAL( KIND = sp_ ), INTENT( IN ) :: A( * )
      REAL( KIND = sp_ ), INTENT( INOUT ) :: B( * )
      REAL( KIND = sp_ ), INTENT( OUT ) :: X( * )
    END SUBROUTINE PARDISO_S

    SUBROUTINE PARDISO_D( PT, MAXFCT, MNUM, MTYPE, PHASE, N, A, IA, JA,        &
                          PERM, NRHS, IPARM, MSGLVL, B, X, ERROR )
      USE MKL_PARDISO_PRIVATE
      USE GALAHAD_KINDS, ONLY: dp_, ip_
      TYPE( MKL_PARDISO_HANDLE ), INTENT( INOUT ) :: PT( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MAXFCT
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MNUM
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MTYPE
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: PHASE
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: N
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: IA( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: JA( * )
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: PERM( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: NRHS
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: IPARM( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MSGLVL
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: ERROR
      REAL( KIND = dp_ ), INTENT( IN ) :: A( * )
      REAL( KIND = dp_ ), INTENT( INOUT ) :: B( * )
      REAL( KIND = dp_ ), INTENT( OUT ) :: X( * )
    END SUBROUTINE PARDISO_D

#ifdef REAL_128
    SUBROUTINE PARDISO_Q( PT, MAXFCT, MNUM, MTYPE, PHASE, N, A, IA, JA,        &
                          PERM, NRHS, IPARM, MSGLVL, B, X, ERROR )
      USE MKL_PARDISO_PRIVATE
      USE GALAHAD_KINDS, ONLY: r16_, ip_
      TYPE( MKL_PARDISO_HANDLE ), INTENT( INOUT ) :: PT( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MAXFCT
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MNUM
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MTYPE
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: PHASE
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: N
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: IA( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: JA( * )
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: PERM( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: NRHS
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: IPARM( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MSGLVL
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: ERROR
      REAL( KIND = r16_ ), INTENT( IN ) :: A( * )
      REAL( KIND = r16_ ), INTENT( INOUT ) :: B( * )
      REAL( KIND = r16_ ), INTENT( OUT ) :: X( * )
    END SUBROUTINE PARDISO_Q
#endif

    SUBROUTINE PARDISO_S_2D( PT, MAXFCT, MNUM, MTYPE, PHASE, N, A, IA, JA,     &
                             PERM, NRHS, IPARM, MSGLVL, B, X, ERROR )
      USE MKL_PARDISO_PRIVATE
      USE GALAHAD_KINDS, ONLY: sp_, ip_
      TYPE( MKL_PARDISO_HANDLE ), INTENT( INOUT ) :: PT( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MAXFCT
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MNUM
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MTYPE
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: PHASE
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: N
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: IA( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: JA( * )
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: PERM( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: NRHS
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: IPARM( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MSGLVL
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: ERROR
      REAL( KIND = sp_ ), INTENT( IN ) :: A( * )
      REAL( KIND = sp_ ), INTENT( INOUT ) :: B( N, * )
      REAL( KIND = sp_ ), INTENT( OUT ) :: X( N, * )
    END SUBROUTINE PARDISO_S_2D

    SUBROUTINE PARDISO_D_2D( PT, MAXFCT, MNUM, MTYPE, PHASE, N, A, IA, JA,     &
                             PERM, NRHS, IPARM, MSGLVL, B, X, ERROR )
      USE MKL_PARDISO_PRIVATE
      USE GALAHAD_KINDS, ONLY: dp_, ip_
      TYPE( MKL_PARDISO_HANDLE ), INTENT( INOUT ) :: PT( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MAXFCT
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MNUM
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MTYPE
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: PHASE
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: N
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: IA( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: JA( * )
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: PERM( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: NRHS
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: IPARM( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MSGLVL
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: ERROR
      REAL( KIND = dp_ ), INTENT( IN ) :: A( * )
      REAL( KIND = dp_ ), INTENT( INOUT ) :: B( N, * )
      REAL( KIND = dp_ ), INTENT( OUT ) :: X( N, * )
    END SUBROUTINE PARDISO_D_2D

#ifdef REAL_128
    SUBROUTINE PARDISO_Q_2D( PT, MAXFCT, MNUM, MTYPE, PHASE, N, A, IA, JA,     &
                             PERM, NRHS, IPARM, MSGLVL, B, X, ERROR )
      USE MKL_PARDISO_PRIVATE
      USE GALAHAD_KINDS, ONLY: r16_, ip_
      TYPE( MKL_PARDISO_HANDLE ), INTENT( INOUT ) :: PT( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MAXFCT
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MNUM
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MTYPE
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: PHASE
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: N
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: IA( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: JA( * )
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: PERM( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: NRHS
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: IPARM( * )
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: MSGLVL
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: ERROR
      REAL( KIND = r16_ ), INTENT( IN ) :: A( * )
      REAL( KIND = r16_ ), INTENT( INOUT ) :: B( N, * )
      REAL( KIND = r16_ ), INTENT( OUT ) :: X( N, * )
    END SUBROUTINE PARDISO_Q_2D
#endif
  END INTERFACE MKL_PARDISO_SOLVE

END MODULE MKL_PARDISO

