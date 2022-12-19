! THIS VERSION: GALAHAD 4.1 - 2022-12-19 AT 12:15 GMT.

#include "galahad_modules.h"

!-*-*-*-*-  G A L A H A D  -  D U M M Y   W S M P   R O U T I N E S  -*-*-*-*-

   SUBROUTINE wsmp_initialize( )
   END SUBROUTINE wsmp_initialize

   SUBROUTINE wsetmaxthrds( numthrds )
   USE GALAHAD_PRECISION
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: numthrds
   END SUBROUTINE wsetmaxthrds

   SUBROUTINE wssmp( n, IA, JA, AVALS, DIAG, PERM, INVP, B, ldb, nrhs,         &
                     AUX, naux, MRP, IPARM, DPARM )
   USE GALAHAD_PRECISION
   USE GALAHAD_SYMBOLS
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, ldb, nrhs, naux
   INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n + 1 ) :: IA
   INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( * ) :: JA
   INTEGER ( KIND = ip_ ), DIMENSION( n ) :: PERM, INVP, MRP
   INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( naux ) :: AUX
   INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( 64 ) :: IPARM
   REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( * ) :: AVALS
   REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( * ) :: DIAG
   REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( ldb, nrhs ) :: B
   REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( 64 ) :: DPARM
   IPARM( 64 ) = GALAHAD_unavailable_option
   END SUBROUTINE wssmp

   SUBROUTINE wsmp_clear( )
   END SUBROUTINE wsmp_clear

   SUBROUTINE wssfree( )
   END SUBROUTINE wssfree

