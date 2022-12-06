! THIS VERSION: 29/03/2021 AT 16:30:00 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   W S M P   R O U T I N E S  -*-*-*-*-

   SUBROUTINE wsmp_initialize( )
   END SUBROUTINE wsmp_initialize

   SUBROUTINE wsetmaxthrds( numthrds )
   INTEGER, INTENT( IN ) :: numthrds
   END SUBROUTINE wsetmaxthrds

   SUBROUTINE wssmp( n, IA, JA, AVALS, DIAG, PERM, INVP, B, ldb, nrhs,         &
                     AUX, naux, MRP, IPARM, DPARM )
   USE GALAHAD_SYMBOLS
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) 
   INTEGER, INTENT( IN ) :: n, ldb, nrhs, naux
   INTEGER, INTENT( INOUT ), DIMENSION( n + 1 ) :: IA
   INTEGER, INTENT( INOUT ), DIMENSION( * ) :: JA
   INTEGER, DIMENSION( n ) :: PERM, INVP, MRP
   INTEGER, INTENT( INOUT ), DIMENSION( naux ) :: AUX
   INTEGER, INTENT( INOUT ), DIMENSION( 64 ) :: IPARM
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( * ) :: AVALS
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( * ) :: DIAG
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( ldb, nrhs ) :: B
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( 64 ) :: DPARM
   IPARM( 64 ) = GALAHAD_unavailable_option
   END SUBROUTINE wssmp

   SUBROUTINE wsmp_clear( )
   END SUBROUTINE wsmp_clear

   SUBROUTINE wssfree( )
   END SUBROUTINE wssfree

