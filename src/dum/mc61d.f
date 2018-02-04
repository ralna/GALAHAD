! THIS VERSION: 21/11/2011 AT 16:00:00 GMT.

!-*-*-  G A L A H A D  -  D U M M Y   M C 6 1    S U B R O U T I N E S  -*-*-

      SUBROUTINE MC61ID(ICNTL,CNTL)
      DOUBLE PRECISION CNTL(5)
      INTEGER ICNTL(10)
      ICNTL(1) = 6
      ICNTL(2) = 6
      ICNTL(3:10) = 0
      CNTL(1:5) = 0.0D+0
      IF ( ICNTL(1) >= 0 ) WRITE(ICNTL(1),                 
     & "( ' We regret that the solution options that you have ', /,
     &  ' chosen are not all freely available with GALAHAD.', /, 
     &  ' If you have HSL (formerly the Harwell Subroutine', /,
     &  ' Library), this option may be enabled by replacing the dummy',
     &   /, ' subroutine MC61ID with its HSL namesake ', /,
     &  ' and dependencies. See ', /,
     &  '   $GALAHAD/src/makedefs/packages for details.' )" )
      END SUBROUTINE MC61ID

      SUBROUTINE MC61AD(JOB,N,LIRN,IRN,ICPTR,PERM,LIW,IW,W,ICNTL,
     &                  CNTL,INFO,RINFO)
      USE GALAHAD_SYMBOLS
      INTEGER JOB,N,LIW,LIRN
      DOUBLE PRECISION RINFO(15)
      DOUBLE PRECISION CNTL(5),W(N)
      INTEGER IRN(LIRN),ICPTR(N+1),INFO(10),ICNTL(10),IW(LIW),PERM(N)
      INFO(1)= GALAHAD_unavailable_option
      IF ( ICNTL(1) >= 0 ) WRITE(ICNTL(1),                 
     & "( ' We regret that the solution options that you have ', /,
     &  ' chosen are not all freely available with GALAHAD.', /,
     &  ' If you have HSL (formerly the Harwell Subroutine', /,
     &  ' Library), this option may be enabled by replacing the dummy',
     &   /, ' subroutine MC61AD with its HSL namesake ', /,
     &  ' and dependencies. See ', /,
     &  '   $GALAHAD/src/makedefs/packages for details.' )" )
      END SUBROUTINE MC61AD

