C  THIS VERSION: GALAHAD 5.0 - 2024-04-29 AT 12:00 GMT.

#include "hsl_subset.h"

      SUBROUTINE MC20AR( nc, maxa, A, INUM, JPTR, JNUM, jdisp )
      USE HSL_KINDS_real, ONLY: ip_, rp_

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER(ip_), INTENT( IN ) :: nc, maxa, jdisp
      INTEGER(ip_), INTENT( INOUT ), DIMENSION( maxa ) :: INUM, JNUM
      INTEGER(ip_), INTENT( OUT ), DIMENSION( nc ) :: JPTR
      REAL(rp_), INTENT( INOUT ), DIMENSION( maxa ) :: A

!  Dummy subroutine available with GALAHAD

      JPTR( 1 ) = - 1_ip_
      RETURN

!  End of dummy subroutine MC20AD

      END SUBROUTINE MC20AR
