      SUBROUTINE MC77ID(ICNTL, CNTL)
      INTEGER LICNTL, LCNTL
      PARAMETER ( LICNTL=10, LCNTL=10 )
      INTEGER ICNTL(LICNTL)
      DOUBLE PRECISION CNTL(LCNTL)
      ICNTL(1) = - 1
!     WRITE( 6,
!    & "( ' We regret that the solution options that you have', /,
!    &    ' chosen are not all freely available with GALAHAD.', /,
!    &    ' If you have HSL (formerly the Harwell Subroutine', /,
!    &    ' Library), this option may be enabled by replacing', /,
!    &    ' the dummy subroutine MC77ID with its HSL namesake ', /,
!    &    ' and dependencies. See ', /,
!    &    '   $GALAHAD/src/makedefs/packages for details.' )" )
      END

      SUBROUTINE MC77AD(JOB,M,N,NNZ,JCST,IRN,A,IW,LIW,DW,LDW,
     &                  ICNTL,CNTL,INFO,RINFO)
      USE GALAHAD_SYMBOLS
      INTEGER LICNTL, LCNTL, LINFO, LRINFO
      PARAMETER ( LICNTL=10, LCNTL=10, LINFO=10, LRINFO=10 )
      INTEGER ICNTL(LICNTL),INFO(LINFO)
      DOUBLE PRECISION CNTL(LCNTL),RINFO(LRINFO)
      INTEGER JOB,M,N,NNZ,LIW,LDW
      INTEGER JCST(N+1),IRN(NNZ),IW(LIW)
      DOUBLE PRECISION A(NNZ),DW(LDW)
      IF ( ICNTL(1) >= 0 ) WRITE( ICNTL(1),
     & "( ' We regret that the solution options that you have', /,
     &    ' chosen are not all freely available with GALAHAD.', /,
     &    ' If you have HSL (formerly the Harwell Subroutine', /,
     &    ' Library), this option may be enabled by replacing', /,
     &    ' the dummy subroutine MC77AD with its HSL namesake ', /,
     &    ' and dependencies. See ', /,
     &    '   $GALAHAD/src/makedefs/packages for details.' )" )
       INFO(1) = GALAHAD_unavailable_option
      END

      SUBROUTINE MC77BD(JOB,M,N,NNZ,IRN,JCN,A,IW,LIW,DW,LDW,
     &                  ICNTL,CNTL,INFO,RINFO)
      USE GALAHAD_SYMBOLS
      INTEGER LICNTL, LCNTL, LINFO, LRINFO
      PARAMETER ( LICNTL=10, LCNTL=10, LINFO=10, LRINFO=10 )
      INTEGER ICNTL(LICNTL),INFO(LINFO)
      DOUBLE PRECISION CNTL(LCNTL),RINFO(LRINFO)
      INTEGER JOB,M,N,NNZ,LIW,LDW
      INTEGER JCN(NNZ),IRN(NNZ),IW(LIW)
      DOUBLE PRECISION A(NNZ),DW(LDW)
      IF ( ICNTL(1) >= 0 ) WRITE( ICNTL(1),
     & "( ' We regret that the solution options that you have', /,
     &    ' chosen are not all freely available with GALAHAD.', /,
     &    ' If you have HSL (formerly the Harwell Subroutine', /,
     &    ' Library), this option may be enabled by replacing', /,
     &    ' the dummy subroutine MC77BD with its HSL namesake ', /,
     &    ' and dependencies. See ', /,
     &    '   $GALAHAD/src/makedefs/packages for details.' )" )
       INFO(1) = GALAHAD_unavailable_option
      END

      SUBROUTINE MC77CD(JOB,M,N,A,LDA,IW,LIW,DW,LDW,
     &                  ICNTL,CNTL,INFO,RINFO)
      USE GALAHAD_SYMBOLS
      INTEGER LICNTL, LCNTL, LINFO, LRINFO
      PARAMETER ( LICNTL=10, LCNTL=10, LINFO=10, LRINFO=10 )
      INTEGER ICNTL(LICNTL),INFO(LINFO)
      DOUBLE PRECISION CNTL(LCNTL),RINFO(LRINFO)
      INTEGER JOB,M,N,LDA,LIW,LDW
      INTEGER IW(LIW)
      DOUBLE PRECISION A(LDA,*),DW(LDW)
      IF ( ICNTL(1) >= 0 ) WRITE( ICNTL(1),
     & "( ' We regret that the solution options that you have', /,
     &    ' chosen are not all freely available with GALAHAD.', /,
     &    ' If you have HSL (formerly the Harwell Subroutine', /,
     &    ' Library), this option may be enabled by replacing', /,
     &    ' the dummy subroutine MC77CD with its HSL namesake ', /,
     &    ' and dependencies. See ', /,
     &    '   $GALAHAD/src/makedefs/packages for details.' )" )
       INFO(1) = GALAHAD_unavailable_option
      END
