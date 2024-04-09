! THIS VERSION: GALAHAD 5.0 - 2024-03-17 AT 11:25 GMT.

#include "hsl_subset.h"

      SUBROUTINE MC77IR( ICNTL, CNTL )
      USE HSL_KINDS_real, ONLY: ip_, rp_
      INTEGER ( KIND = ip_ ) :: LICNTL, LCNTL
      PARAMETER ( LICNTL = 10, LCNTL = 10 )
      INTEGER ( KIND = ip_ ) :: ICNTL( LICNTL )
      REAL ( KIND = rp_ ) :: CNTL( LCNTL )
      ICNTL( 1 ) = - 1
!     WRITE( 6,
!    & "( ' We regret that the solution options that you have', /,
!    &    ' chosen are not all freely available with GALAHAD.', /,
!    &    ' If you have HSL (formerly the Harwell Subroutine', /,
!    &    ' Library), this option may be enabled by replacing', /,
!    &    ' the dummy subroutine MC77IR with its HSL namesake ', /,
!    &    ' and dependencies. See ', /,
!    &    '   $GALAHAD/src/makedefs/packages for details.' )" )
      END SUBROUTINE MC77IR

      SUBROUTINE MC77AR( job, m, n, nnz, JCST, IRN, A, IW,
     &                   liw, DW, ldw, ICNTL, CNTL, INFO, RINFO )
      USE HSL_KINDS_real, ONLY: ip_, rp_

      INTEGER ( KIND = ip_ ) :: licntl, lcntl, linfo, lrinfo
      PARAMETER ( licntl = 10, lcntl = 10, linfo = 10, lrinfo = 10 )
      INTEGER ( KIND = ip_ ) :: ICNTL( licntl ), INFO( linfo )
      REAL ( KIND = rp_ ) :: CNTL( lcntl ), RINFO( lrinfo )
      INTEGER ( KIND = ip_ ) :: job, m, n, nnz, liw, ldw
      INTEGER ( KIND = ip_ ) :: JCST( n + 1 ), IRN( nnz ), IW( liw )
      REAL ( KIND = rp_ ) :: A( nnz ), DW( ldw )
      IF ( ICNTL( 1 ) >= 0 ) WRITE( ICNTL( 1 ),
     & "( ' We regret that the solution options that you have', /,
     &    ' chosen are not all freely available with GALAHAD.', /,
     &    ' If you have HSL (formerly the Harwell Subroutine', /,
     &    ' Library), this option may be enabled by replacing', /,
     &    ' the dummy subroutine MC77AR with its HSL namesake ', /,
     &    ' and dependencies. See ', /,
     &    '   $GALAHAD/src/makedefs/packages for details.' )" )
       INFO( 1 ) = - 29
      END SUBROUTINE MC77AR

      SUBROUTINE MC77BR( job, m, n, nnz, IRN, JCN, A, IW,
     &                   liw, DW, ldw, ICNTL, CNTL, INFO, RINFO )
      USE HSL_KINDS_real, ONLY: ip_, rp_
      INTEGER ( KIND = ip_ ) :: licntl, lcntl, linfo, lrinfo
      PARAMETER ( licntl = 10, lcntl = 10, linfo = 10, lrinfo = 10 )
      INTEGER ( KIND = ip_ ) :: ICNTL( licntl ), INFO( linfo )
      REAL ( KIND = rp_ ) :: CNTL( lcntl ), RINFO( lrinfo )
      INTEGER ( KIND = ip_ ) :: job, m, n, nnz, liw, ldw
      INTEGER ( KIND = ip_ ) :: JCN( nnz ), IRN( nnz ), IW( liw )
      REAL ( KIND = rp_ ) :: A( nnz ), DW( ldw )
      IF ( ICNTL( 1 ) >= 0 ) WRITE( ICNTL( 1 ),
     & "( ' We regret that the solution options that you have', /,
     &    ' chosen are not all freely available with GALAHAD.', /,
     &    ' If you have HSL (formerly the Harwell Subroutine', /,
     &    ' Library), this option may be enabled by replacing', /,
     &    ' the dummy subroutine MC77BR with its HSL namesake ', /,
     &    ' and dependencies. See ', /,
     &    '   $GALAHAD/src/makedefs/packages for details.' )" )
       INFO( 1 ) = - 29
      END SUBROUTINE MC77BR

      SUBROUTINE MC77CR( job, m, n, A, lda, IW, liw, DW, ldw,
     &                   ICNTL, CNTL, INFO, RINFO )
      USE HSL_KINDS_real, ONLY: ip_, rp_
      INTEGER ( KIND = ip_ ) :: licntl, lcntl, linfo, lrinfo
      PARAMETER ( licntl = 10, lcntl = 10, linfo = 10, lrinfo = 10 )
      INTEGER ( KIND = ip_ ) :: ICNTL( licntl ), INFO( linfo )
      REAL ( KIND = rp_ ) :: CNTL( lcntl ), RINFO( lrinfo )
      INTEGER ( KIND = ip_ ) :: job, m, n, lda, liw, ldw
      INTEGER ( KIND = ip_ ) :: IW( liw )
      REAL ( KIND = rp_ ) :: A( lda, *), DW( ldw )
      IF ( ICNTL( 1 ) >= 0 ) WRITE( ICNTL( 1 ),
     & "( ' We regret that the solution options that you have', /,
     &    ' chosen are not all freely available with GALAHAD.', /,
     &    ' If you have HSL (formerly the Harwell Subroutine', /,
     &    ' Library), this option may be enabled by replacing', /,
     &    ' the dummy subroutine MC77CR with its HSL namesake ', /,
     &    ' and dependencies. See ', /,
     &    '   $GALAHAD/src/makedefs/packages for details.' )" )
       INFO( 1 ) = - 29
      END SUBROUTINE MC77CR
