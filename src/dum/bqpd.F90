! THIS VERSION: GALAHAD 5.6 - 2026-07-26 AT 13:40 GMT.

#include "galahad_modules.h"

!-*-*-*-  G A L A H A D  -  D U M M Y   B Q P D     S U B R O U T I N E  -*-*-*-

  SUBROUTINE BQPD( n, m, k, kmax, A, LA, X, BL, BU, f, fmin, G, R, W, E, LS,   &
                   ALP, LP, mlp, peq, WS, LWS, mode, ifail, INFO, iprint, nout )
  USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
  IMPLICIT NONE
  INTEGER ( KIND = ip_ ) :: n, m, k, kmax, mlp, peq, mode, ifail, iprint, nout
  REAL ( KIND = rp_ ) :: f, fmin
  INTEGER ( KIND = ip_ ), DIMENSION( * ) :: LA, LS, LP, LWS, INFO
  REAL ( KIND = rp_ ), DIMENSION( * ) :: A, X, BL, BU, G, R, E, ALP, W, WS
  ifail =  - 199 ! error code
  END SUBROUTINE BQPD
