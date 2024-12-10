! THIS VERSION: GALAHAD 5.1 - 2024-11-18 AT 14:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-  G A L A H A D    N O R M S   M O D U L E  -*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould
!
!  History -
!   originally released pre GALAHAD Version 2.0. May 22nd 2004

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_NORMS_precision

      USE GALAHAD_KINDS_precision
      USE GALAHAD_BLAS_inter_precision, ONLY : NRM2

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: NRM2, ONE_NORM, TWO_NORM, INFINITY_NORM

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_

    CONTAINS

!-*-*-*-*-  G A L A H A D   O N E  _ N O R M   F U N C T I O N   -*-*-*-*-

       FUNCTION ONE_NORM( X )

!  Compute the l_1 norm of the vector X

!  Dummy arguments

       REAL ( KIND = rp_ ) :: ONE_NORM
       REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( : ) :: X

!  Local variable

       INTEGER ( KIND = ip_ ) :: n
       n = SIZE( X )

       IF ( n > 0 ) THEN
         ONE_NORM = SUM( ABS( X ) )
       ELSE
         ONE_NORM = zero
       END IF
       RETURN

!  End of function ONE_NORM

       END FUNCTION ONE_NORM

!-*-*-*-*-  G A L A H A D   T W O  _ N O R M   F U N C T I O N   -*-*-*-*-

       FUNCTION TWO_NORM( X )

!  Compute the l_2 norm of the vector X

!  Dummy arguments

       REAL ( KIND = rp_ ) :: TWO_NORM
       REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( : ) :: X

!  Local variable

       INTEGER ( KIND = ip_ ) :: n
       n = SIZE( X )

       IF ( n > 0 ) THEN
         TWO_NORM = NRM2( n, X, 1_ip_ )
       ELSE
         TWO_NORM = zero
       END IF
       RETURN

!  End of function TWO_NORM

       END FUNCTION TWO_NORM

!-*-*-*-  G A L A H A D   I N F I N I T Y  _ N O R M   F U N C T I O N   -*-*-*-

       FUNCTION INFINITY_NORM( X )

!  Compute the l_infinity norm of the vector X

!  Dummy arguments

       REAL ( KIND = rp_ ) :: INFINITY_NORM
       REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( : ) :: X

!  Local variable

       INTEGER ( KIND = ip_ ) :: n
       n = SIZE( X )

       IF ( n > 0 ) THEN
         INFINITY_NORM = MAXVAL( ABS( X ) )
       ELSE
         INFINITY_NORM = zero
       END IF
       RETURN

!  End of function INFINITY_NORM

       END FUNCTION INFINITY_NORM

!  End of module GALAHAD_NORMS

    END MODULE GALAHAD_NORMS_precision
