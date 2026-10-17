! THIS VERSION: GALAHAD 5.6 - 2026-06-23 AT 11:30 GMT.

#include "galahad_modules.h"

!-*-*-  G A L A H A D  -  D U M M Y   B P M P D X    S U B R O U T I N E  -*-*-

  SUBROUTINE BPMPDX( m, n, nz, qn, qnz, acolcnt, acolidx, acolnzs,             &
                     qcolcnt, qcolidx, qcolnzs, rhs, obj, lbound,              &
                     ubound, primal, dual, status, big, code, opt,             &
                     msizi, msizr, f, loglevel, iter )
  USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
  IMPLICIT NONE
  INTEGER ( KIND = ip_ ), INTENT( IN ) :: m, n, nz, qn, qnz, loglevel
  INTEGER ( KIND = ip_ ), INTENT( OUT ) :: code, iter
  INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: msizi, msizr
  INTEGER ( KIND = ip_ ), INTENT( IN ) :: acolcnt( n ), acolidx( nz )
  INTEGER ( KIND = ip_ ), INTENT( IN ) :: qcolcnt( n ), qcolidx( qnz )
  INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status( n + m )
  REAL ( KIND = rp_ ), INTENT( IN ) :: big, f
  REAL ( KIND = rp_ ), INTENT( OUT ) :: opt
  REAL ( KIND = rp_ ), INTENT( IN ) :: acolnzs( nz ), qcolnzs( qnz )
  REAL ( KIND = rp_ ), INTENT( IN ) :: rhs( m ), obj( n )
  REAL ( KIND = rp_ ), INTENT( IN ) :: lbound( n + m ), ubound( n + m )
  REAL ( KIND = rp_ ), INTENT( INOUT ) :: primal( n + m ), dual( n + m )
  code =  - 199 ! error code
  END SUBROUTINE BPMPDX




          
