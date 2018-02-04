   SUBROUTINE pardisoinit( PT, mtype, solver, IPARM, DPARM, error )
   USE GALAHAD_SYMBOLS
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )  
!  INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )
!  INTEGER ( KIND = long ), INTENT( INOUT ), DIMENSION( 64 ) :: PT
   INTEGER, INTENT( INOUT ), DIMENSION( 64 ) :: PT
   INTEGER, INTENT( IN ) :: mtype, solver
   INTEGER, INTENT( INOUT ), DIMENSION( 64 ) :: IPARM
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( 64 ) :: DPARM
   INTEGER, INTENT( OUT ) :: error
   error = GALAHAD_unavailable_option
   END SUBROUTINE pardisoinit

   SUBROUTINE pardiso( PT, maxfct, mnum, mtype, phase, n, A, IA, JA,           &
                       PERM, nrhs, IPARM, msglvl, B, X, error, DPARM )
   USE GALAHAD_SYMBOLS
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )  
!  INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )
!  INTEGER ( KIND = long ), INTENT( INOUT ), DIMENSION( 64 ) :: PT
   INTEGER, INTENT( INOUT ), DIMENSION( 64 ) :: PT
   INTEGER, INTENT( IN ) :: maxfct, mnum, mtype, phase, n, nrhs, msglvl
   INTEGER, INTENT( OUT ) :: error
   INTEGER, INTENT( INOUT ), DIMENSION( 64 ) :: IPARM
   INTEGER, INTENT( IN ), DIMENSION( n ) :: PERM
   INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: IA
   INTEGER, INTENT( IN ), DIMENSION( IA( n + 1 ) - 1 ) :: JA
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( IA( n + 1 ) - 1 ) :: A
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n , nrhs ) :: X
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n , nrhs ) :: B
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( 64 ) :: DPARM
   error = GALAHAD_unavailable_option
   END SUBROUTINE pardiso
