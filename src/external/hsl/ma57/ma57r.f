C  THIS VERSION: GALAHAD 5.0 - 2024-07-07 AT 09:00 GMT.

#include "hsl_subset.h"

      SUBROUTINE MA57IR( CNTL, ICNTL )
      USE HSL_KINDS_real, ONLY: ip_, rp_
      REAL ( KIND = rp_ ) :: CNTL(5)
      INTEGER (ip_) :: ICNTL(20)
      END SUBROUTINE MA57IR

      SUBROUTINE MA57AR( N, NE, IRN, JCN, LKEEP, KEEP, IWORK,
     *                   ICNTL, INFO, RINFO )
      USE HSL_KINDS_real, ONLY: ip_, rp_
      INTEGER ( KIND = ip_ ) :: N, NE, IRN(NE), JCN(NE), IWORK(5*N)
      INTEGER ( KIND = ip_ ) :: LKEEP, KEEP(LKEEP), ICNTL(20),INFO(40)
      REAL ( KIND = rp_ ) :: RINFO(20)
      END SUBROUTINE MA57AR

      SUBROUTINE MA57BR( N, NE, A, FACT, LFACT, IFACT, LIFACT,
     *                   LKEEP, KEEP, PPOS, ICNTL, CNTL, INFO, RINFO )
      USE HSL_KINDS_real, ONLY: ip_, rp_
      INTEGER ( KIND = ip_ ) :: N, NE, LFACT, LIFACT, LKEEP
      REAL ( KIND = rp_ ) :: A(NE), FACT(LFACT)
      REAL ( KIND = rp_ ) :: RINFO(20)
      REAL ( KIND = rp_ ) :: CNTL(5)
      INTEGER ( KIND = ip_ ) :: ICNTL(20), IFACT(LIFACT)
      INTEGER ( KIND = ip_ ) ::   INFO(40), KEEP(LKEEP), PPOS(N)
      END SUBROUTINE MA57BR

      SUBROUTINE MA57CR( JOB, N, FACT, LFACT, IFACT, LIFACT,
     *                   NRHS, RHS, LRHS, W, LW, IW1, ICNTL, INFO )
      USE HSL_KINDS_real, ONLY: ip_, rp_
      INTEGER ( KIND = ip_ ) :: JOB, N, LFACT
      REAL ( KIND = rp_ ) :: FACT(LFACT)
      INTEGER ( KIND = ip_ ) :: LIFACT, IFACT(LIFACT), NRHS, LRHS, LW
      REAL ( KIND = rp_ ) :: W(LW), RHS(LRHS,NRHS)
      INTEGER ( KIND = ip_ ) :: IW1(N), ICNTL(20), INFO(40)
      END SUBROUTINE MA57CR

      SUBROUTINE MA57DR( JOB, N, NE, A, IRN, JCN, FACT,
     *                   LFACT, IFACT, LIFACT, RHS, X, RESID,
     *                   W, IW, ICNTL, CNTL, INFO, RINFO )
      USE HSL_KINDS_real, ONLY: ip_, rp_
      INTEGER ( KIND = ip_ ) :: JOB, N, NE,LFACT, LIFACT
      REAL ( KIND = rp_ ) :: A(NE)
      INTEGER ( KIND = ip_ ) :: IRN(NE), JCN(NE)
      REAL ( KIND = rp_ ) :: FACT(LFACT)
      INTEGER ( KIND = ip_ ) :: IFACT(LIFACT)
      REAL ( KIND = rp_ ) :: RHS(N), X(N), RESID(N), W(N,*)
      INTEGER ( KIND = ip_ ) :: IW(N), ICNTL(20)
      REAL ( KIND = rp_ ) :: CNTL(5)
      INTEGER ( KIND = ip_ ) :: INFO(40)
      REAL ( KIND = rp_ ) :: RINFO(20)
      END SUBROUTINE MA57DR

      SUBROUTINE MA57ER( N, IC, KEEP, FACT, LFACT, NEWFAC,
     *                   LNEW, IFACT, LIFACT, NEWIFC, LINEW, INFO )
      USE HSL_KINDS_real, ONLY: ip_, rp_
      INTEGER ( KIND = ip_ ) :: N, IC, KEEP(*), LFACT, LNEW
      INTEGER ( KIND = ip_ ) :: LIFACT,LINEW, INFO(40)
      REAL ( KIND = rp_ ) :: FACT(LFACT), NEWFAC(LNEW)
      INTEGER ( KIND = ip_ ) :: IFACT(LIFACT), NEWIFC(LINEW)
      END SUBROUTINE MA57ER
