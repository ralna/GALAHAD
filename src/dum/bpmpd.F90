! THIS VERSION: GALAHAD 5.6 - 2026-06-23 AT 11:30 GMT.

#include "galahad_modules.h"

!-*-*-  G A L A H A D  -  D U M M Y   B P M P D     S U B R O U T I N E  -*-*-

  SUBROUTINE bpmpd( m, n, nz, qn, qnz, acolcnt, acolidx, acolnzs,              &
                    qcolcnt, qcolidx, qcolnzs, rhs, obj, lbound,               &
                    ubound, primal, dual, status, big, code, opt,              &
                    msizi, msizr )
  USE GALAHAD_KINDS_precision
  IMPLICIT NONE
  INTEGER ( KIND = ip_ ), INTENT( IN ) :: m, n, nz, qn, qnz
  INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: code, msizi, msizr
  INTEGER ( KIND = ip_ ), INTENT( IN ) :: acolcnt( n ), acolidx( nz )
  INTEGER ( KIND = ip_ ), INTENT( IN ) :: qcolcnt( n ), qcolidx( qnz )
  INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status( n + m )
  REAL ( KIND = rp_ ), INTENT( IN ) :: big
  REAL ( KIND = rp_ ), INTENT( OUT ) :: opt
  REAL ( KIND = rp_ ), INTENT( IN ) :: acolnzs( nz ), qcolnzs( qnz )
  REAL ( KIND = rp_ ), INTENT( IN ) :: rhs( m ), obj( n )
  REAL ( KIND = rp_ ), INTENT( IN ) :: lbound( n + m ), ubound( n + m )
  REAL ( KIND = rp_ ), INTENT( INOUT ) :: primal( n + m ), dual( n + m )
  code =  - 199 ! error code
  END SUBROUTINE bpmpd




          
