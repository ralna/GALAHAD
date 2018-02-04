* *******************************************************************
* COPYRIGHT (c) 1987 Hyprotech UK
* All rights reserved.
*
* None of the comments in this Copyright notice between the lines
* of asterisks shall be removed or altered in any way.
*
* This Package is intended for compilation without modification,
* so most of the embedded comments have been removed.
*
* ALL USE IS SUBJECT TO LICENCE. For full details of an HSL ARCHIVE
* Licence, see http://hsl.rl.ac.uk/archive/cou.html
*
* Please note that for an HSL ARCHIVE Licence:
*
* 1. The Package must not be copied for use by any other person.
*    Supply of any part of the library by the Licensee to a third party
*    shall be subject to prior written agreement between AEA
*    Hyprotech UK Limited and the Licensee on suitable terms and
*    conditions, which will include financial conditions.
* 2. All information on the Package is provided to the Licensee on the
*    understanding that the details thereof are confidential.
* 3. All publications issued by the Licensee that include results obtained
*    with the help of one or more of the Packages shall acknowledge the
*    use of the Packages. The Licensee will notify the Numerical Analysis
*    Group at Rutherford Appleton Laboratory of any such publication.
* 4. The Packages may be modified by or on behalf of the Licensee
*    for such use in research applications but at no time shall such
*    Packages or modifications thereof become the property of the
*    Licensee. The Licensee shall make available free of charge to the
*    copyright holder for any purpose all information relating to
*    any modification.
* 5. Neither CCLRC nor Hyprotech UK Limited shall be liable for any
*    direct or consequential loss or damage whatsoever arising out of
*    the use of Packages by the Licensee.
* *******************************************************************
*
*######DATE 4 Oct 1992
C       Toolpack tool decs employed.
*######DATE 10 Feb 1993 modified.
C       Calls to FA04A/BS -> FA04A/B.
C
      SUBROUTINE YM01A(M,N,NZIN,NZOUT,IBW,ITYPE,NONSIN,IRN,A,JCOLST,IW)
      INTEGER IBW,ITYPE,M,N,NZIN,NZOUT
      LOGICAL NONSIN
      REAL A(*)
      INTEGER IRN(*),IW(*),JCOLST(*)
      INTEGER I,II,III,IL,IRECT,ISW,ITYP,IU,J,JB,JBEG,JL,JLEN,JUSED,K,
     +        KOL,L,LENCOL,MAXMN,MINMN,NCOL,NZ,NZERO
      LOGICAL BAND,SYM,UNSYM,YESA
      REAL FA04A
      EXTERNAL FA04A
      EXTERNAL FA04B,YM01B
      INTRINSIC IABS,MAX0,MIN0
      YESA = ITYPE .GE. 0
      ITYP = IABS(ITYPE)
      IF (ITYP.EQ.0 .OR. ITYP.GT.2) ITYP = 1
      IF (M.NE.N) ITYP = 1
      SYM = ITYP .EQ. 2
      UNSYM = ITYP .EQ. 1
      MINMN = MIN0(M,N)
      IF (MINMN.LE.0) GO TO 170
      MAXMN = MAX0(M,N)
      BAND = IBW .LT. MAXMN .AND. IBW .GE. 0
      IF (BAND) THEN
        IRECT = MAXMN - MINMN
        IF (IRECT.GT.0) THEN
          IF (IBW.LE.MINMN) NZERO = (IBW+1)* (MINMN-IBW) +
     +                              IBW* (IBW+1)/2
          IF (IBW.GT.MINMN) NZERO = MINMN* (MINMN+1)/2
          IF (IBW.GT.IRECT) NZERO = NZERO + MINMN*IBW +
     +                              (IBW-IRECT)* (IBW-IRECT+1)/2
          IF (IBW.LE.IRECT) NZERO = NZERO + MINMN*IBW
        ELSE
          NZERO = (IBW+1)* (2*MINMN-IBW)/2
          IF (UNSYM) NZERO = NZERO + IBW* (2*MINMN-IBW-1)/2
        END IF
      ELSE
        IF (SYM) NZERO = N* (N+1)/2
        IF (.NOT.SYM) NZERO = M*N
      END IF
      IF (NZIN.GE.NZERO) GO TO 120
      IF (NONSIN .AND. .NOT.BAND .AND. M.LT.N) CALL YM01B(M,N,JCOLST,IW)
      DO 10 I = 1,N
        IW(I) = 0
   10 CONTINUE
      IF (NONSIN) THEN
        IF (BAND .OR. M.GE.N) THEN
          DO 20 I = 1,MINMN
            IW(I) = 1
   20     CONTINUE
        ELSE
          DO 30 I = 1,M
            IW(JCOLST(I)) = 1
   30     CONTINUE
        END IF
      END IF
      NZ = NZIN
      IF (NONSIN) NZ = NZ - MINMN
      JLEN = M
      JL = JLEN
      JB = M
      IF (BAND) THEN
        JB = IBW + 1
        IF (.NOT.SYM) JB = 2*IBW + 1
      END IF
      IF (NONSIN) JB = JB - 1
      DO 50 K = 1,NZ
   40   CALL FA04B(N,J)
        IF (BAND) THEN
          IF (J.GT.M+IBW) GO TO 40
          IF (.NOT.SYM) JLEN = MIN0(J+IBW,M) - MAX0(1,J-IBW) + 1
          IF (SYM) JLEN = MIN0(J+IBW,M) - J + 1
          JL = JLEN
          IF (NONSIN .AND. J.LE.M) JL = JL - 1
        ELSE
          IF (SYM) THEN
            JLEN = M - J + 1
            JL = JLEN
            IF (NONSIN) JL = JL - 1
          END IF
        END IF
        IF (BAND .OR. SYM) THEN
          CALL FA04B(JB,L)
          IF (L.GT.JL) GO TO 40
          IF (IW(J).EQ.JLEN) GO TO 40
          JUSED = IW(J)
          IF (NONSIN .AND. J.LE.M) JUSED = JUSED - 1
          CALL FA04B(JL,L)
          IF (L.LE.JUSED) GO TO 40
        END IF
        IF (IW(J).EQ.JLEN) GO TO 40
        IW(J) = IW(J) + 1
   50 CONTINUE
      IF (NONSIN .AND. .NOT.BAND .AND. M.LT.N) THEN
        DO 60 K = 1,M
          J = JCOLST(K)
          IW(J) = -IW(J)
   60   CONTINUE
      END IF
      IF (NONSIN .AND. UNSYM .AND. .NOT.BAND) CALL YM01B(MINMN,M,JCOLST,
     +    IW(N+1))
      DO 70 K = 1,M
        IW(N+K) = 0
   70 CONTINUE
      K = 0
      KOL = 0
      DO 100 J = 1,N
        LENCOL = IABS(IW(J)+0)
        JBEG = K + 1
        ISW = 0
        IF (NONSIN) THEN
          IF (SYM .OR. BAND) THEN
            IF (J.LE.M) THEN
              K = K + 1
              IRN(K) = J
              IW(N+J) = J
              LENCOL = LENCOL - 1
            END IF
          ELSE
            IF ((M.LT.N.AND.IW(J).LT.0) .OR. M.GE.N) THEN
              ISW = 1
              K = K + 1
              IF (M.LT.N) THEN
                KOL = KOL + 1
                IRN(K) = JCOLST(KOL)
              ELSE
                IRN(K) = JCOLST(J)
              END IF
              IW(N+IRN(K)) = J
              LENCOL = LENCOL - 1
            END IF
          END IF
        END IF
        IF (LENCOL.EQ.0) GO TO 100
        IL = 1
        IF (BAND) IL = MAX0(J-IBW,1)
        IF (SYM) IL = J
        IU = M
        IF (BAND) IU = MIN0(J+IBW,M)
        NCOL = IU - IL + 1
        DO 90 III = 1,LENCOL
   80     CALL FA04B(NCOL,II)
          I = IL - 1 + II
          IF (IW(N+I).EQ.J) GO TO 80
          K = K + 1
          IRN(K) = I
          IW(N+I) = J
   90   CONTINUE
        IF (ISW.EQ.1) THEN
          CALL FA04B(LENCOL,L)
          ISW = IRN(JBEG)
          IRN(JBEG) = IRN(JBEG+L-1)
          IRN(JBEG+L-1) = ISW
        END IF
  100 CONTINUE
      JCOLST(1) = 1
      DO 110 J = 1,N
        JCOLST(J+1) = JCOLST(J) + IABS(IW(J)+0)
  110 CONTINUE
      GO TO 150
  120 K = 1
      DO 140 J = 1,N
        JCOLST(J) = K
        IL = 1
        IF (BAND) IL = MAX0(1,J-IBW)
        IF (SYM) IL = J
        IU = M
        IF (BAND) IU = MIN0(M,J+IBW)
        DO 130 I = IL,IU
          IRN(K) = I
          K = K + 1
  130   CONTINUE
  140 CONTINUE
      JCOLST(N+1) = K
  150 NZOUT = JCOLST(N+1) - 1
      IF (.NOT.YESA) GO TO 170
      DO 160 K = 1,NZOUT
        A(K) = FA04A(-1)
  160 CONTINUE
  170 RETURN
      END
      SUBROUTINE YM01B(N,M,PERM,INVPRM)
      INTEGER M,N
      INTEGER INVPRM(M),PERM(N)
      INTEGER I,J,NN
      EXTERNAL FA04B
      DO 10 I = 1,M
        INVPRM(I) = 0
   10 CONTINUE
      NN = N
      IF (M.EQ.N) NN = N - 1
      DO 30 I = 1,NN
   20   CALL FA04B(M,J)
        IF (INVPRM(J).NE.0) GO TO 20
        INVPRM(J) = I
        PERM(I) = J
   30 CONTINUE
      IF (NN.EQ.N) GO TO 50
      DO 40 I = 1,N
        IF (INVPRM(I).NE.0) GO TO 40
        PERM(N) = I
        INVPRM(I) = N
        GO TO 50
   40 CONTINUE
   50 RETURN
      END
