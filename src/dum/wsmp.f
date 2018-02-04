   SUBROUTINE wsmp_initialize( )
   END SUBROUTINE wsmp_initialize

   SUBROUTINE wsmp_clear( )
   END SUBROUTINE wsmp_clear

   SUBROUTINE wsffree( )
   END SUBROUTINE wsffree

   SUBROUTINE wsafree( )
   END SUBROUTINE wsafree

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

   SUBROUTINE wscalz( n, IA, JA, OPTIONS, PERM, INVP, nnzl, wspace,            &
                      AUX, naux, info )
   USE GALAHAD_SYMBOLS
   INTEGER, INTENT( IN ) :: n, naux
   INTEGER :: nnzl, wspace, info
   INTEGER, INTENT( INOUT ), DIMENSION( n + 1 ) :: IA
   INTEGER, INTENT( INOUT ), DIMENSION( * ) :: JA
   INTEGER, INTENT( INOUT ), DIMENSION( 5 ) :: OPTIONS
   INTEGER, DIMENSION( n ) :: PERM, INVP
   INTEGER, INTENT( INOUT ), DIMENSION( naux ) :: AUX
   info = GALAHAD_unavailable_option
   END SUBROUTINE wscalz

   SUBROUTINE wscchf( n, IA, JA, AVALS, PERM, INVP, AUX, naux, info )
   USE GALAHAD_SYMBOLS
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) 
   INTEGER, INTENT( IN ) :: n, naux
   INTEGER :: info
   INTEGER, INTENT( INOUT ), DIMENSION( n + 1 ) :: IA
   INTEGER, INTENT( INOUT ), DIMENSION( * ) :: JA
   INTEGER, DIMENSION( n ) :: PERM, INVP
   INTEGER, INTENT( INOUT ), DIMENSION( naux ) :: AUX
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( * ) :: AVALS
   info = GALAHAD_unavailable_option
   END SUBROUTINE wscchf

   SUBROUTINE wscldl( n, IA, JA, AVALS, PERM, INVP, AUX, naux, info )
   USE GALAHAD_SYMBOLS
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) 
   INTEGER, INTENT( IN ) :: n, naux
   INTEGER :: info
   INTEGER, INTENT( INOUT ), DIMENSION( n + 1 ) :: IA
   INTEGER, INTENT( INOUT ), DIMENSION( * ) :: JA
   INTEGER, DIMENSION( n ) :: PERM, INVP
   INTEGER, INTENT( INOUT ), DIMENSION( naux ) :: AUX
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( * ) :: AVALS
   info = GALAHAD_unavailable_option
   END SUBROUTINE wscldl

   SUBROUTINE wsslv( n, PERM, INVP, B, ldb, nrhs, niter, AUX, naux )
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) 
   INTEGER, INTENT( IN ) :: n, ldb, nrhs, niter, naux
   INTEGER, DIMENSION( n ) :: PERM, INVP
   INTEGER, INTENT( INOUT ), DIMENSION( naux ) :: AUX
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( ldb, nrhs ) :: B
   END SUBROUTINE wsslv

   SUBROUTINE wscsvx( n, IA, JA, AVALS, PERM, INVP, B, ldb, nrhs, AUX, naux,   &
                      rcond, info )
   USE GALAHAD_SYMBOLS
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) 
   INTEGER, INTENT( IN ) :: n, ldb, nrhs, naux
   INTEGER :: info
   REAL ( KIND = wp ) :: rcond
   INTEGER, INTENT( INOUT ), DIMENSION( n + 1 ) :: IA
   INTEGER, INTENT( INOUT ), DIMENSION( * ) :: JA
   INTEGER, DIMENSION( n ) :: PERM, INVP
   INTEGER, INTENT( INOUT ), DIMENSION( naux ) :: AUX
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( * ) :: AVALS
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( ldb, nrhs ) :: B
   info = GALAHAD_unavailable_option
   END SUBROUTINE wscsvx

   SUBROUTINE wsetmaxthrds( numthrds )
   INTEGER, INTENT( IN ) :: numthrds
   END SUBROUTINE wsetmaxthrds
