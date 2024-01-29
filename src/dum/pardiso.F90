! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-  G A L A H A D  -  D U M M Y   P A R D I S O  R O U T I N E S  -*-*-*-

   SUBROUTINE PARDISOINIT( PT, mtype, solver, IPARM, DPARM, error )
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SYMBOLS
   INTEGER ( KIND = long_ ), INTENT( INOUT ), DIMENSION( 64 ) :: PT
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: mtype, solver
   INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( 64 ) :: IPARM
   REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( 64 ) :: DPARM
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: error
   error = GALAHAD_unavailable_option
   END SUBROUTINE PARDISOINIT

   SUBROUTINE PARDISO( PT, maxfct, mnum, mtype, phase, n, A, IA, JA,           &
                       PERM, nrhs, IPARM, msglvl, B, X, error, DPARM )
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SYMBOLS
   INTEGER ( KIND = long_ ), INTENT( INOUT ), DIMENSION( 64 ) :: PT
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: maxfct, mnum, mtype, phase
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nrhs, msglvl
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: error
   INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( 64 ) :: IPARM
   INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n ) :: PERM
   INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n + 1 ) :: IA
   INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( IA( n + 1 ) - 1 ) :: JA
   REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( IA( n + 1 ) - 1 ) :: A
   REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n , nrhs ) :: X
   REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n , nrhs ) :: B
   REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( 64 ) :: DPARM
   error = GALAHAD_unavailable_option
   END SUBROUTINE PARDISO
