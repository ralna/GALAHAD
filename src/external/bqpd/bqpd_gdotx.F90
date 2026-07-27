! THIS VERSION: GALAHAD 5.6 - 2026-07-27 AT 10:20 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*- G A L A H A D  -  B Q P D    S U B R O U T I N E S -*-*-*-*-*-*-

    SUBROUTINE gdotx( n, X, WS, LWS, V )
    USE GALAHAD_KINDS_precision, ONLY: ip_, rp_

    IMPLICIT NONE

!  given H and x, form v = H x

    INTEGER ( ip_ ), INTENT( IN ) :: n
    INTEGER ( ip_ ), INTENT( IN ), DIMENSION( 0 : * ) :: LWS
    REAL ( rp_ ), INTENT( IN ), DIMENSION( n ) :: X
    REAL ( rp_ ), INTENT( IN ), DIMENSION( * ) :: WS
    REAL ( rp_ ), INTENT( OUT ), DIMENSION( n ) :: V

    INTEGER ( ip_ ) :: i, ij, j, ng
    REAL ( rp_ ) :: a, b

!  special case for tridiagonal Toeplitz H

    IF ( LWS( 0 ) == 0 ) THEN
      a = WS( 1 )
      b = WS( 2 )
      V( 1 ) = a * X( 1 ) + b * x( 2 )
      DO i = 2, n - 1
         v(i) = a * X( i ) + b * ( X( i - 1 ) + X( i + 1 ) )
      END DO
      V( n ) = b * X( n - 1 ) + a * X( n )

! normal case where the upper triangle of H is in co-ordinate format

    ELSE
      DO i = 1, n
        V( i ) =  0.0_rp_
      enddo
      ng = LWS( 0 )
!     WRITE(6,*) ' new product'
      DO ij = 1,ng
        i = LWS( ij )
        j = LWS( ng + ij )
!       WRITE(6,*) ' i, j, val ', i, j, WS( ij )
        V( i ) = v( i ) + WS( ij ) * X( j )
        IF ( i /= j ) V( j ) = V( j ) + WS( ij ) * X( i )
      END DO
!     WRITE(6,*) 'v  = ', V
    END IF
    RETURN
    END SUBROUTINE gdotx
