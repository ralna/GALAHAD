! THIS VERSION: GALAHAD 5.6 - 2026-06-23 AT 11:30 GMT.

#include "galahad_modules.h"

!-*-*-  G A L A H A D  -  D U M M Y   E 0 4 N Q F    S U B R O U T I N E  -*-*-

  SUBROUTINE E04NQF( start, qphx, m, n, ne, nname, lenc, ncolh, iobj, objadd,  &
                     prob, acol, inda, loca, bl, bu, c, names, helast, hs,     &
                     x, pi, rc, ns, ninf, sinf, obj, cw, lencw, iw, leniw,     &
                     rw, lenrw, cuser, iuser, ruser, ifail )
  USE GALAHAD_KINDS_precision
  IMPLICIT NONE
  INTEGER ( KIND = ip_ ), INTENT( IN ) :: m, n, ne, nname, lenc, ncolh, iobj
  INTEGER ( KIND = ip_ ), INTENT( IN ) :: inda( ne ), loca( n + 1 )
  INTEGER ( KIND = ip_ ), INTENT( IN ) :: helast( n + m ), lencw, leniw, lenrw
  INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: hs( n + m ), ns, iw( leniw )
  INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iuser( * ), ifail
  INTEGER ( KIND = ip_ ), INTENT( OUT ) :: ninf
  REAL ( KIND = rp_ ), INTENT( IN ) :: objadd
  REAL ( KIND = rp_ ), INTENT( INOUT ) :: acol( ne ), bl( n + m ), bu( n + m )
  REAL ( KIND = rp_ ), INTENT( INOUT ) :: c( MAX( 1, lenc ) ), x( n + m )
  REAL ( KIND = rp_ ), INTENT( INOUT ) :: rw( lenrw ), ruser( * )
  REAL ( KIND = rp_ ), INTENT( OUT ) :: pi( m ), rc( n + m ), sinf, obj
  CHARACTER ( LEN = 1 ), INTENT ( IN ) :: start
  CHARACTER ( LEN = 8 ), INTENT ( IN ) :: prob, names( nname )
  CHARACTER ( LEN = 8 ), INTENT ( INOUT ) :: cw( lencw ), cuser( * )
  INTERFACE
    SUBROUTINE qphx( ncolh, x, hx, nstate, cuser, iuser, ruser )
    USE GALAHAD_KINDS_precision
    IMPLICIT NONE
    INTEGER ( KIND = ip_ ), INTENT( IN ) :: ncolh, nstate
    INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iuser( * )
    REAL ( KIND = rp_ ), INTENT( IN ) :: x(ncolh)
    REAL ( KIND = rp_ ), INTENT( INOUT ) :: ruser( * )
    REAL ( KIND = rp_ ), INTENT( OUT ) :: hx(ncolh)
    CHARACTER ( LEN = 8 ), INTENT( INOUT) :: cuser(*)
    END SUBROUTINE qphx
  END INTERFACE
  ifail = - 199  ! error code
  END SUBROUTINE E04NQF

!-*-*-  G A L A H A D  -  D U M M Y   E 0 4 N P F    S U B R O U T I N E  -*-*-

  SUBROUTINE E04NPF( cw, lencw, iw, leniw, rw, lenrw, ifail )
  USE GALAHAD_KINDS_precision
  IMPLICIT NONE
  INTEGER ( KIND = ip_ ), INTENT( IN ) :: lencw, leniw, lenrw
  INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: ifail
  INTEGER ( KIND = ip_ ), INTENT( OUT ) :: iw( leniw )
  REAL ( KIND = rp_ ), INTENT( OUT ) :: rw( lenrw )
  CHARACTER ( LEN = 8 ), INTENT( OUT ) :: cw( lencw )
  ifail = - 199  ! error code
  END SUBROUTINE E04NPF

!-*-*-  G A L A H A D  -  D U M M Y   E 0 4 N R F    S U B R O U T I N E  -*-*-

  SUBROUTINE E04NRF( ispecs, cw, iw, rw, ifail )
  USE GALAHAD_KINDS_precision
  IMPLICIT NONE
  INTEGER ( KIND = ip_ ), INTENT( IN ) :: ispecs
  INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iw( * ),  ifail
  REAL ( KIND  = rp_ ), INTENT( INOUT ) :: rw( * )
  CHARACTER ( LEN = 8 ), INTENT( INOUT ) :: cw( * )
  ifail = - 199  ! error code
  END SUBROUTINE E04NRF
