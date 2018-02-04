* *******************************************************************
* COPYRIGHT (c) 1979 Hyprotech UK and
* Council for the Central Laboratory of the Research Councils
* All rights reserved.
*
* None of the comments in this Copyright notice between the lines
* of asterisks shall be removed or altered in any way.
*
* This Package is intended for compilation without modification,
* so most of the embedded comments have been removed.
*
* ALL USE IS SUBJECT TO LICENCE. For full details of the ACADEMIC
* SOFTWARE LICENCE, see http://hsl.rl.ac.uk/hsl2007/cou/academic.html
*
* Please note that for an ACADEMIC Licence:
*
* 1. The Packages may only be used for academic research or teaching
*    purposes by the Licensee, and must not be copied by the Licensee for
*    use by any other persons. Use of the Packages in any commercial
*    application shall be subject to prior written agreement between
*    Hyprotech UK Limited and the Licensee on suitable terms and
*    conditions, which will include financial conditions.
* 2. All information on the Package is provided to the Licensee on the
*    understanding that the details thereof are confidential.
* 3. All publications issued by the Licensee that include results obtained
*    with the help of one or more of the Packages shall acknowledge the
*    use of the Packages. The Licensee will notify the Numerical Analysis
*    Group at Rutherford Appleton Laboratory (STFC) of any such publication.
* 4. The Packages may be modified by or on behalf of the Licensee
*    for such use in research applications but at no time shall such
*    Packages or modifications thereof become the property of the
*    Licensee. The Licensee shall make available free of charge to the
*    copyright holder for any purpose all information relating to
*    any modification.
* 5. Neither STFC nor Hyprotech UK Limited shall be liable for any
*    direct or consequential loss or damage whatsoever arising out of
*    the use of Packages by the Licensee.
* *******************************************************************
*
C Original date 5 June 2001
C  June 2001: threadsafe version of MA31
C 20/2/02 Cosmetic changes applied to reduce single/double differences
C 12/7/04 LFULL and LFULDD in MA61C made real to avoid integer overflow

C 12th July 2004 Version 1.0.0. Version numbering added.

C 20th December 2005 Version 1.1.0. LFULDD and LFULL declared REAL in
C    MA61CD.

      SUBROUTINE MA61I(ICNTL,CNTL,KEEP)
      REAL CNTL(3)
      INTEGER ICNTL(5)
      INTEGER KEEP(12)
      INTEGER I
      ICNTL(1) = 6
      ICNTL(2) = 6
      ICNTL(3) = 0
      ICNTL(4) = 0
      ICNTL(5) = 0
      CNTL(1) = 0.8
      CNTL(2) = 0.0
      CNTL(3) = 0.0
      DO 10 I = 1,12
         KEEP(I) = 0
   10 CONTINUE
      RETURN
      END
      SUBROUTINE MA61A(N,NZ,A,INI,INJ,IAI,IAJ,IK,IW,W,C,ICNTL,CNTL,INFO,
     +                 KEEP)
      REAL ZERO,ONE
      PARAMETER (ZERO=0.0E0,ONE=1.0E0)
      REAL C
      INTEGER IAI,IAJ,N,NZ
      REAL A(IAJ),W(N,3)
      INTEGER IK(N,4),INI(IAI),INJ(IAJ),IW(N,4)
      REAL CNTL(3)
      INTEGER KEEP(12),ICNTL(5),INFO(10)
      INTEGER I,IAJNZ,II,IR,J,K,KI,KJ,KK,KL,KLL,KP,KPP,KR,NM1,NZ0,NZP1
      INTEGER IPD,LCOL,LROW,NCP,ND,NUAL,NUCL,NURL,IFLAG
      EXTERNAL MA61C,MA61E
      INTRINSIC SQRT
      NURL = KEEP(1)
      NUCL = KEEP(2)
      NUAL = KEEP(3)
      LROW = KEEP(4)
      LCOL = KEEP(5)
      NCP = KEEP(6)
      ND = KEEP(7)
      IPD = KEEP(8)
      IF (N.LT.1) GO TO 220
      IF (NZ.LT.N) GO TO 230
      IF (IAI.LT.NZ) GO TO 240
      IF (IAJ.LT.2*NZ) GO TO 250
      DO 20 I = 1,N
         DO 10 J = 1,3
            W(I,J) = ZERO
            IK(I,J) = 0
   10    CONTINUE
   20 CONTINUE
      NUAL = 0
      IFLAG = 0
      ND = 0
      NCP = 0
      DO 60 K = 1,NZ
         I = INI(K)
         J = INJ(K)
         IF (I.LT.1 .OR. I.GT.N) GO TO 260
         IF (J.LT.I .OR. J.GT.N) GO TO 260
         IF (I.EQ.J) GO TO 40
         IF (A(K).NE.ZERO) GO TO 30
         ND = ND + 1
         GO TO 60
   30    CONTINUE
         IK(I,1) = IK(I,1) + 1
         IK(J,2) = IK(J,2) + 1
         NUAL = NUAL + 1
         A(NUAL) = A(K)
         INI(NUAL) = I
         INJ(NUAL) = J
         GO TO 60
   40    CONTINUE
         ND = ND + 1
         IF (W(I,1).EQ.ZERO) GO TO 50
         IF (ICNTL(2).GT.0) WRITE (ICNTL(2),FMT=9000) I
 9000    FORMAT (/,/,' Warning: more than one diagonal entry in row',I5)
         IFLAG = 1
   50    CONTINUE
         W(I,1) = W(I,1) + A(K)
   60 CONTINUE
      NZ0 = NZ - ND
      LCOL = NZ0
      LROW = NZ0
      IF (NZ0.EQ.0) GO TO 290
      KJ = IAI - NZ0 + 1
      KI = 1
      DO 70 I = 1,N
         KI = KI + IK(I,1)
         KJ = KJ + IK(I,2)
         IW(I,1) = KI
         IW(I,2) = KJ
         IF (W(I,1).LE.ZERO) GO TO 270
         W(I,1) = ONE/SQRT(W(I,1))
   70 CONTINUE
      CALL MA61E(INI,INJ,NZ0,IW,N,A)
      KK = 0
      DO 120 IR = 1,N
         KPP = IW(IR,1)
         IW(IR,1) = KK + 1
         KLL = KPP + IK(IR,1) - 1
         IF (KPP.GT.KLL) GO TO 120
         DO 90 K = KPP,KLL
            J = INJ(K)
            IF (W(J,3).EQ.ZERO) GO TO 80
            IF (ICNTL(2).GT.0) WRITE (ICNTL(2),FMT=9010) IR,J
 9010       FORMAT (/,/,' Warning: there is more than one entry in row',
     +             I5,' and column',I5)
            IFLAG = 1
   80       CONTINUE
            W(J,3) = W(J,3) + A(K)
   90    CONTINUE
         DO 110 K = KPP,KLL
            J = INJ(K)
            IF (W(J,3).EQ.ZERO) GO TO 100
            KK = KK + 1
            A(KK) = W(J,3)*W(IR,1)*W(J,1)
            INJ(KK) = J
            W(J,3) = ZERO
            KR = IW(J,2) - 1
            IW(J,2) = KR
            INI(KR) = IR
            GO TO 110
  100       CONTINUE
            ND = ND + 1
            LROW = LROW - 1
            LCOL = LCOL - 1
            IK(IR,1) = IK(IR,1) - 1
            IK(J,2) = IK(J,2) - 1
  110    CONTINUE
  120 CONTINUE
      IF (IFLAG.NE.1) GO TO 150
      NZ0 = NZ - ND
      NM1 = N - 1
      DO 140 I = 1,NM1
         KP = IW(I,2) + IK(I,2)
         KL = IW(I+1,2) - 1
         IF (KP.GT.KL) GO TO 140
         DO 130 K = KP,KL
            INI(K) = 0
  130    CONTINUE
  140 CONTINUE
  150 CONTINUE
      NUAL = IAJ + 1
      DO 180 II = 1,N
         IK(II,4) = IK(II,1)
         I = N - II + 1
         W(I,2) = ONE
         KP = IW(I,1)
         KL = KP + IK(I,1) - 1
         IF (KP.GT.KL) GO TO 170
         DO 160 KK = KP,KL
            K = KP + KL - KK
            NUAL = NUAL - 1
            A(NUAL) = A(K)
            INJ(NUAL) = INJ(K)
  160    CONTINUE
  170    CONTINUE
         IW(I,1) = NUAL - NZ0
  180 CONTINUE
      NURL = 0
      NZP1 = NZ0 + 1
      NUCL = IW(1,2)
      NUAL = NUAL - NZ0
      IAJNZ = IAJ - NZ0
      KEEP(1) = NURL
      KEEP(2) = NUCL
      KEEP(3) = NUAL
      KEEP(4) = LROW
      KEEP(5) = LCOL
      KEEP(6) = NCP
      KEEP(8) = IPD
      CALL MA61C(N,NZ0,W(1,2),A(NZP1),INI,INJ(NZP1),IAI,IAJNZ,IK,IW,
     +           IW(1,3),W(1,3),IFLAG,C,ICNTL,CNTL,KEEP)
      NURL = KEEP(1)
      NUCL = KEEP(2)
      NUAL = KEEP(3)
      LROW = KEEP(4)
      LCOL = KEEP(5)
      NCP = KEEP(6)
      IPD = KEEP(8)
      KP = 1
      DO 210 I = 1,N
         KL = KP + IK(I,4) - 1
         IF (KP.GT.KL) GO TO 200
         DO 190 K = KP,KL
            INI(K) = I
  190    CONTINUE
  200    CONTINUE
         KP = KL + 1
  210 CONTINUE
      GO TO 310
  220 CONTINUE
      IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=9020)
 9020 FORMAT (/,/,' Error return from MA61A because',
     +    ' N is out of range.')
      IFLAG = -1
      GO TO 280
  230 CONTINUE
      IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=9030)
 9030 FORMAT (/,/,' Error return from MA61A because',
     +   ' NZ is out of range.')
      IFLAG = -2
      GO TO 280
  240 CONTINUE
      IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=9040)
 9040 FORMAT (/,/,' Error return from MA61A because',
     +   ' IAI is out of range.')
      IFLAG = -3
      GO TO 280
  250 CONTINUE
      IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=9050)
 9050 FORMAT (/,/,' Error return from MA61A because',
     +   ' IAJ is out of range.')
      IFLAG = -4
      GO TO 280
  260 CONTINUE
      IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=9060) K,I,J
 9060 FORMAT (/,/,' Error return from MA61A because',
     +   ' element',I7,' is in row',I5,' and column',I5)
      IFLAG = -5
      GO TO 280
  270 CONTINUE
      IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=9070) I
 9070 FORMAT (/,/,' Error return from MA61A because diagonal element',
     +   I5,' is zero or negative')
      IFLAG = -6
  280 CONTINUE
      GO TO 310
  290 CONTINUE
      DO 300 I = 1,N
         IK(I,2) = I
         IK(I,1) = 0
         IK(I,3) = 0
         W(I,1) = ONE/SQRT(W(I,1))
         W(I,2) = ONE
  300 CONTINUE
  310 CONTINUE
      KEEP(1) = NURL
      KEEP(2) = NUCL
      KEEP(3) = NUAL
      KEEP(4) = LROW
      KEEP(5) = LCOL
      KEEP(6) = NCP
      KEEP(7) = ND
      KEEP(8) = IPD
      KEEP(9) = IFLAG
      INFO(1) = KEEP(9)
      INFO(2) = KEEP(4)
      INFO(3) = KEEP(5)
      INFO(4) = KEEP(6)
      INFO(5) = KEEP(7)
      INFO(6) = KEEP(8)
      RETURN
      END
      SUBROUTINE MA61C(N,NZ,D,A,INI,INJ,IAI,IAJ,IK,IP,IW,W,IFLAG,C,
     +                 ICNTL,CNTL,KEEP)
      REAL ZERO,ONE
      PARAMETER (ZERO=0.0E0,ONE=1.0E0)
      REAL EPSTOL,U,CMAX
      PARAMETER (EPSTOL=2.0E-6,U=1.0E2,CMAX=1.0E20)
      REAL C
      INTEGER IAI,IAJ,IFLAG,N,NZ
      REAL A(IAJ),D(N),W(N)
      INTEGER IK(N,3),INI(IAI),INJ(IAJ),IP(N,2),IW(N,2)
      REAL CNTL(3)
      INTEGER ICNTL(5),KEEP(12)
      REAL AA,AL,ALFA,B1,B2,C0,DJP,LFULDD,LFULL,PFILL,PIVT
      INTEGER I,II,IIP,IIPP1,IL,IN,IP1,IPDP1,IR,J,J1,JJ,JP,K,KC,KK,KL,
     +        KL1,KL2,KLC,KLJ,KLL,KLR,KP,KP2,KPC,KPI,KPJ,KPP,KPR,KR,KRL,
     +        KS,L,MCL,NC,NFILL,NM1,NR,NRJP,NRM1,NZ0,NZC,
     +        NZI
      INTEGER IPD,LCOL,LROW,NCP,NUAL,NUCL,NURL
      LOGICAL CHANGE
      EXTERNAL MA61D
      INTRINSIC ABS,MAX,INT,SQRT,REAL
      NURL = KEEP(1)
      NUCL = KEEP(2)
      NUAL = KEEP(3)
      LROW = KEEP(4)
      LCOL = KEEP(5)
      NCP = KEEP(6)
      IPD = KEEP(8)
      CHANGE = .TRUE.
      IF (C.LE.ZERO) CHANGE = .FALSE.
      NZ0 = NZ
      IPD = N
      ALFA = 1.0/0.90
      B1 = -.03
      B2 = .03
      NFILL = IAJ - NZ0 - N
      MCL = LCOL
      C0 = ABS(C)
      C0 = C0/N
      C = C**2
      DO 10 I = 1,N
         D(I) = D(I) + C0
         IK(I,3) = 0
   10 CONTINUE
      DO 20 I = 1,N
         NZI = IK(I,1) + IK(I,2) + 1
         IN = IK(NZI,3)
         IK(NZI,3) = I
         IW(I,2) = IN
         IW(I,1) = 0
         IF (IN.NE.0) IW(IN,1) = I
   20 CONTINUE
      DO 530 IIP = 1,N
         DO 30 NRJP = 1,N
            JP = IK(NRJP,3)
            IF (JP.GT.0) GO TO 40
   30    CONTINUE
   40    CONTINUE
         DO 100 L = 1,2
            KPP = IP(JP,L)
            KLL = IK(JP,L) + KPP - 1
            IF (KPP.GT.KLL) GO TO 100
            DO 90 K = KPP,KLL
               IF (L.EQ.2) GO TO 50
               J = INJ(K)
               GO TO 60
   50          CONTINUE
               J = INI(K)
   60          CONTINUE
               IL = IW(J,1)
               IN = IW(J,2)
               IW(J,2) = -1
               IF (IN.LT.0) GO TO 90
               IF (IL.EQ.0) GO TO 70
               IW(IL,2) = IN
               GO TO 80
   70          CONTINUE
               NZ = IK(J,1) + IK(J,2) + 1
               IK(NZ,3) = IN
   80          CONTINUE
               IF (IN.GT.0) IW(IN,1) = IL
   90       CONTINUE
  100    CONTINUE
         IL = IW(JP,1)
         IN = IW(JP,2)
         IW(JP,2) = -10
         IF (IN.LT.0) GO TO 110
         NZ = IK(JP,1) + IK(JP,2) + 1
         IK(NZ,3) = IN
         IF (IN.GT.0) IW(IN,1) = IL
  110    CONTINUE
         IW(JP,1) = -IIP
         IF (LROW+IK(JP,1)+IK(JP,2).GT.IAJ-N) C = CMAX
         IF (NURL+IK(JP,1)+IK(JP,2).LT.NUAL) GO TO 120
         CALL MA61D(A,INJ,IAJ,N,IK,IP,.TRUE.,NCP,NUCL,NUAL)
  120    CONTINUE
         KP = IP(JP,1)
         KL = IK(JP,1) + KP - 1
         IP(JP,1) = NURL + 1
         IF (KP.GT.KL) GO TO 180
         DO 170 K = KP,KL
            J = INJ(K)
            KPC = IP(J,2)
            NZ = IK(J,2) - 1
            IK(J,2) = NZ
            KLC = KPC + NZ
            IF (KLC.GT.KPC) GO TO 130
            INI(KPC) = 0
            GO TO 160
  130       CONTINUE
            DO 140 KC = KPC,KLC
               IF (JP.EQ.INI(KC)) GO TO 150
  140       CONTINUE
  150       CONTINUE
            INI(KC) = INI(KLC)
            INI(KLC) = 0
  160       CONTINUE
            LCOL = LCOL - 1
            NURL = NURL + 1
            INJ(NURL) = J
            A(NURL) = A(K)
            INJ(K) = 0
  170    CONTINUE
  180    CONTINUE
         KP2 = IP(JP,2)
         KL2 = IK(JP,2) + KP2 - 1
         IF (KP2.GT.KL2) GO TO 220
         DO 210 K = KP2,KL2
            NURL = NURL + 1
            LCOL = LCOL - 1
            I = INI(K)
            KPR = IP(I,1)
            KLR = KPR + IK(I,1) - 1
            DO 190 KR = KPR,KLR
               IF (JP.EQ.INJ(KR)) GO TO 200
  190       CONTINUE
  200       CONTINUE
            INJ(KR) = INJ(KLR)
            A(NURL) = A(KR)
            A(KR) = A(KLR)
            INJ(KLR) = 0
            IK(I,1) = IK(I,1) - 1
            INJ(NURL) = I
            INI(K) = 0
  210    CONTINUE
  220    CONTINUE
         NZC = IK(JP,1) + IK(JP,2)
         IK(JP,1) = NZC
         IK(JP,2) = 0
         KP = IP(JP,1)
         KL = KP + NZC - 1
         C0 = EPSTOL*U
         IF (KP.GT.KL) GO TO 240
         DO 230 K = KP,KL
            AA = A(K)
            C0 = MAX(C0,ABS(AA))
            J = INJ(K)
            W(J) = AA
  230    CONTINUE
  240    CONTINUE
         DJP = D(JP)
         IF (DJP.GT.C0/U) GO TO 250
         IFLAG = 2
         IF (ICNTL(2).GT.0) WRITE (ICNTL(2),FMT=9000) JP
 9000    FORMAT (/,/,'  Warning: modification of zero or negative',
     +          ' diagonal entry has been performed in location',I7)
         D(JP) = C0
         IF (C0.EQ.EPSTOL*U) D(JP) = ONE
  250    CONTINUE
         IF (KP.GT.KL) GO TO 520
         DO 480 NC = 1,NZC
            KC = IP(JP,1) + NC - 1
            IR = INJ(KC)
            AL = A(KC)/D(JP)
            IF (LROW+IK(IR,1)+IK(JP,1).GT.IAJ-N) C = CMAX
            IF (NURL+IK(IR,1)+IK(JP,1).LT.NUAL) GO TO 260
            CALL MA61D(A,INJ,IAJ,N,IK,IP,.TRUE.,NCP,NUCL,NUAL)
  260       CONTINUE
            KR = IP(IR,1)
            KRL = KR + IK(IR,1) - 1
            IF (KR.GT.KRL) GO TO 280
            DO 270 KS = KR,KRL
               J = INJ(KS)
               IF (IW(J,2).NE.-1) GO TO 270
               IW(J,2) = 1
               A(KS) = A(KS) - AL*W(J)
  270       CONTINUE
  280       CONTINUE
            DO 470 KS = KP,KL
               J = INJ(KS)
               IF (J.LT.IR) GO TO 460
               IF (IW(J,2).EQ.1) GO TO 460
               AA = -AL*W(J)
               IF (IR.NE.J) GO TO 290
               D(IR) = D(IR) + AA
               GO TO 460
  290          CONTINUE
               IF (AA*AA.GT.C*ABS(D(IR)*D(J))) GO TO 300
               D(J) = D(J) + AA
               D(IR) = D(IR) + AA
               GO TO 460
  300          CONTINUE
               LROW = LROW + 1
               IK(IR,1) = IK(IR,1) + 1
               IF (KR.GT.KRL) GO TO 360
               IF (KRL.EQ.IAJ) GO TO 310
               IF (INJ(KRL+1).NE.0) GO TO 310
               KRL = KRL + 1
               INJ(KRL) = J
               A(KRL) = AA
               GO TO 370
  310          CONTINUE
               IF (KR.NE.NUAL) GO TO 320
               NUAL = NUAL - 1
               GO TO 330
  320          CONTINUE
               IF (INJ(KR-1).NE.0) GO TO 340
  330          CONTINUE
               KR = KR - 1
               IP(IR,1) = KR
               INJ(KR) = J
               A(KR) = AA
               GO TO 370
  340          CONTINUE
               DO 350 KK = KR,KRL
                  NUAL = NUAL - 1
                  INJ(NUAL) = INJ(KK)
                  A(NUAL) = A(KK)
                  INJ(KK) = 0
  350          CONTINUE
  360          CONTINUE
               NUAL = NUAL - 1
               INJ(NUAL) = J
               A(NUAL) = AA
               IP(IR,1) = NUAL
               KR = NUAL
               KRL = KR + IK(IR,1) - 1
  370          CONTINUE
               NZ = IK(J,2)
               K = IP(J,2)
               KL1 = K + NZ - 1
               LCOL = LCOL + 1
               IF (NZ.EQ.0) GO TO 410
               IF (KL1.EQ.IAI) GO TO 380
               IF (INI(KL1+1).NE.0) GO TO 380
               INI(KL1+1) = IR
               GO TO 450
  380          CONTINUE
               IF (K.NE.NUCL) GO TO 390
               IF (NUCL.EQ.1) GO TO 410
               NUCL = NUCL - 1
               GO TO 400
  390          CONTINUE
               IF (INI(K-1).NE.0) GO TO 410
  400          CONTINUE
               K = K - 1
               INI(K) = IR
               IP(J,2) = K
               GO TO 450
  410          CONTINUE
               IF (NZ+1.LT.NUCL) GO TO 420
               IF (LCOL+NZ+2.GE.IAI) C = CMAX
               CALL MA61D(A,INI,IAI,N,IK(1,2),IP(1,2),.FALSE.,NCP,NUCL,
     +                    NUAL)
               K = IP(J,2)
               KL1 = K + NZ - 1
  420          CONTINUE
               IF (K.GT.KL1) GO TO 440
               DO 430 KK = K,KL1
                  NUCL = NUCL - 1
                  INI(NUCL) = INI(KK)
                  INI(KK) = 0
  430          CONTINUE
  440          CONTINUE
               NUCL = NUCL - 1
               INI(NUCL) = IR
               IP(J,2) = NUCL
  450          CONTINUE
               IK(J,2) = NZ + 1
  460          CONTINUE
               IW(J,2) = -1
  470       CONTINUE
  480    CONTINUE
         DO 490 K = KP,KL
            J = INJ(K)
            W(J) = 0.
            A(K) = A(K)/D(JP)
            NZ = IK(J,1) + IK(J,2) + 1
            IN = IK(NZ,3)
            IW(J,2) = IN
            IW(J,1) = 0
            IK(NZ,3) = J
            IF (IN.NE.0) IW(IN,1) = J
  490    CONTINUE
         MCL = MAX(MCL,LCOL)
         PIVT = REAL(IIP)/REAL(N)
         IF (C.NE.CMAX) GO TO 500
         IF (IPD.LT.IIP) GO TO 520
         IPD = IIP
         IF (PIVT.GT..9) GO TO 520
         IFLAG = 4
         IF (ICNTL(2).GT.0) WRITE (ICNTL(2),FMT=9010) IIP
         GO TO 520
 9010    FORMAT (/,/,' Warning: available space used at pivot step',I7)
  500    CONTINUE
         IF (.NOT.CHANGE) GO TO 520
         PFILL = REAL(LROW-NZ0)/REAL(NFILL)
         IF (PIVT.GT.0.9) GO TO 520
         IF (PFILL.LT.ALFA*PIVT+B1) GO TO 510
         IF (PFILL.LT.ALFA*PIVT+B2) GO TO 520
         C = 2.25*C
  510    CONTINUE
         ALFA = (1.0-PFILL)/ (0.9-PIVT)
         B1 = PFILL - PIVT*ALFA - 0.03
         B2 = B1 + 0.06
  520    CONTINUE
         NR = N - IIP
         LFULL = NR* (NR-1.0)/2.0
         LFULDD = CNTL(1)*LFULL
         IF (LCOL.GE.LFULDD .AND. NURL+LFULL.LT.IAJ) GO TO 540
  530 CONTINUE
  540 CONTINUE
      IPD = IIP
      C = SQRT(C)
      LCOL = MCL
      IF (.NOT.CHANGE) C = -C
      KK = 0
      DO 580 I = 1,NR
         JP = IK(I,3)
  550    CONTINUE
         IF (JP.LE.0) GO TO 570
         KK = KK + 1
         INI(KK) = JP
         JP = IW(JP,2)
         GO TO 550
  570    CONTINUE
         IF (KK.EQ.NR) GO TO 590
  580 CONTINUE
  590 CONTINUE
      IF (NR.EQ.1) GO TO 620
      NRM1 = NR - 1
      DO 610 I = 1,NRM1
         J1 = I + 1
         DO 600 J = J1,NR
            IF (INI(J).GT.INI(I)) GO TO 600
            JJ = INI(I)
            INI(I) = INI(J)
            INI(J) = JJ
  600    CONTINUE
  610 CONTINUE
  620 CONTINUE
      DO 630 I = 1,NR
         II = INI(I)
         IW(II,1) = - (IPD+I)
  630 CONTINUE
      DO 640 I = 1,N
         IR = -IW(I,1)
         IK(IR,2) = I
  640 CONTINUE
      IPDP1 = IPD + 1
      NM1 = N - 1
      IF (IPDP1.GT.NM1) GO TO 740
      DO 690 IIP = IPDP1,NM1
         JP = IK(IIP,2)
         KP = IP(JP,1)
         KL = KP + IK(JP,1) - 1
         IF (KP.GT.KL) GO TO 660
         DO 650 K = KP,KL
            J = INJ(K)
            INJ(K) = 0
            W(J) = A(K)
  650    CONTINUE
  660    CONTINUE
         IF (NURL+N-IIP.LT.NUAL) GO TO 670
         CALL MA61D(A,INJ,IAJ,N,IK,IP,.TRUE.,NCP,NUCL,NUAL)
  670    CONTINUE
         IP(JP,1) = NURL + 1
         IK(JP,1) = N - IIP
         IIPP1 = IIP + 1
         DO 680 I = IIPP1,N
            J = IK(I,2)
            NURL = NURL + 1
            A(NURL) = W(J)
            INJ(NURL) = J
            W(J) = ZERO
  680    CONTINUE
  690 CONTINUE
      LROW = NURL
      DO 730 IIP = IPDP1,NM1
         JP = IK(IIP,2)
         KPI = IP(JP,1)
         IP1 = IIP + 1
         IF (IP1.EQ.N) GO TO 720
         DO 710 J = IP1,NM1
            JJ = IK(J,2)
            KPJ = IP(JJ,1)
            KLJ = KPJ + IK(JJ,1) - 1
            AL = A(KPI)/D(JP)
            D(JJ) = D(JJ) - AL*A(KPI)
            KK = KPI + 1
            DO 700 K = KPJ,KLJ
               A(K) = A(K) - AL*A(KK)
               KK = KK + 1
  700       CONTINUE
            A(KPI) = AL
            KPI = KPI + 1
  710    CONTINUE
  720    CONTINUE
         JJ = IK(N,2)
         AL = A(KPI)/D(JP)
         D(JJ) = D(JJ) - AL*A(KPI)
         A(KPI) = AL
  730 CONTINUE
  740 CONTINUE
      KEEP(1) = NURL
      KEEP(2) = NUCL
      KEEP(3) = NUAL
      KEEP(4) = LROW
      KEEP(5) = LCOL
      KEEP(6) = NCP
      KEEP(8) = IPD
      RETURN
      END
      SUBROUTINE MA61D(A,IRN,IA,N,IK,IP,ROW,NCP,NUCL,NUAL)
      INTEGER IA,N,NCP,NUCL,NUAL
      LOGICAL ROW
      REAL A(IA)
      INTEGER IK(N),IP(N),IRN(IA)
      INTEGER IPI,J,K,KK,KL,KN,NN,NZ
      NCP = NCP + 1
      DO 10 J = 1,N
         NZ = IK(J)
         NN = NUAL
         IF (.NOT.ROW) NN = NUCL
         IF (NZ.LE.0 .OR. IP(J).LT.NN) GO TO 10
         K = IP(J)
         IK(J) = IRN(K)
         IRN(K) = -J
   10 CONTINUE
      KN = IA + 1
      IPI = IA + 1
      KL = IA - NUCL + 1
      IF (ROW) KL = IA - NUAL + 1
      DO 30 KK = 1,KL
         K = IA + 1 - KK
         IF (IRN(K).EQ.0) GO TO 30
         KN = KN - 1
         IF (ROW) A(KN) = A(K)
         IF (IRN(K).GE.0) GO TO 20
         J = -IRN(K)
         IRN(K) = IK(J)
         IP(J) = KN
         IK(J) = IPI - KN
         IPI = KN
   20    CONTINUE
         IRN(KN) = IRN(K)
   30 CONTINUE
      IF (ROW) GO TO 40
      NUCL = KN
      GO TO 50
   40 CONTINUE
      NUAL = KN
   50 CONTINUE
      RETURN
      END
      SUBROUTINE MA61E(IN1,IN2,NZ,IP,N,A)
      INTEGER N,NZ
      REAL A(NZ)
      INTEGER IN1(NZ),IN2(NZ),IP(N)
      REAL A1,A2
      INTEGER I,IC1,IC2,IDUMMY,IR1,IR2,KI
      DO 40 I = 1,NZ
         IR1 = IN1(I)
         IF (IR1.LT.0) GO TO 40
         IC1 = IN2(I)
         A1 = A(I)
         KI = IP(IR1) - 1
         DO 10 IDUMMY = 1,NZ
            IF (I.EQ.KI) GO TO 20
            IR2 = IN1(KI)
            IC2 = IN2(KI)
            IN1(KI) = -IR1
            IN2(KI) = IC1
            IP(IR1) = KI
            IR1 = IR2
            IC1 = IC2
            A2 = A(KI)
            A(KI) = A1
            A1 = A2
            KI = IP(IR1) - 1
   10    CONTINUE
   20    CONTINUE
         IF (IDUMMY.EQ.1) GO TO 30
         A(KI) = A1
         IN2(KI) = IC1
         IN1(KI) = -IR1
   30    CONTINUE
         IP(IR1) = I
   40 CONTINUE
      RETURN
      END
      SUBROUTINE MA61G(N,A,INJ,IAJ,D,IK,B,LROW)
      INTEGER IAJ,N,LROW
      REAL A(IAJ),B(N),D(N)
      INTEGER IK(N,2),INJ(IAJ)
      REAL BIC,BIR
      INTEGER IC,IIP,IPI,IR,K,KL,KP
      KP = 1
      DO 30 IIP = 1,N
         IC = IK(IIP,2)
         KL = KP + IK(IC,1) - 1
         BIC = B(IC)
         IF (KP.GT.KL) GO TO 20
         DO 10 K = KP,KL
            IR = INJ(K)
            B(IR) = B(IR) - A(K)*BIC
   10    CONTINUE
   20    CONTINUE
         KP = KL + 1
   30 CONTINUE
      KL = LROW
      DO 60 IPI = 1,N
         IIP = N + 1 - IPI
         IR = IK(IIP,2)
         BIR = 0.
         KP = KL - IK(IR,1) + 1
         IF (KP.GT.KL) GO TO 50
         DO 40 K = KP,KL
            IC = INJ(K)
            BIR = BIR - A(K)*B(IC)
   40    CONTINUE
   50    CONTINUE
         B(IR) = B(IR)/D(IR) + BIR
         KL = KP - 1
   60 CONTINUE
      RETURN
      END
      SUBROUTINE MA61B(N,NZ,A,INI,INJ,IAJ,IK,B,W,W1,KMAX,EPS,ICNTL,INFO,
     +                 KEEP)
      INTEGER IAJ,N,NZ
      REAL A(*),B(*),EPS(2),W(N,3),W1(N,3)
      INTEGER IK(N,3),INI(*),INJ(*),KMAX(2)
      INTEGER ICNTL(5),INFO(10),KEEP(12)
      INTEGER I,IFJ,NZ0,NZP1,LROW,ND,IFLAG
      EXTERNAL MA61F
      LROW = KEEP(4)
      ND = KEEP(7)
      IFLAG = KEEP(9)
      IF (IFLAG.LT.0) GO TO 60
      IF (N.LT.1) GO TO 30
      IF (NZ.LT.N) GO TO 40
      IF (IAJ.LT.2*NZ) GO TO 50
      NZ0 = NZ - ND
      NZP1 = NZ0 + 1
      IFJ = IAJ - NZ0
      KMAX(2) = KMAX(1)
      EPS(2) = EPS(1)
      DO 10 I = 1,N
         B(I) = B(I)*W(I,1)
   10 CONTINUE
      CALL MA61F(N,NZ0,A,INI,INJ,IFJ,A(NZP1),W(1,2),INJ(NZP1),IK,B,
     +           W(1,3),W1,W1(1,2),W1(1,3),KMAX(2),EPS(2),LROW)
      DO 20 I = 1,N
         B(I) = B(I)*W(I,1)
   20 CONTINUE
      IF (EPS(2).LE.EPS(1)) GO TO 80
      IFLAG = 3
      IF (ICNTL(2).GT.0) WRITE (ICNTL(2),FMT=9000) KMAX(1)
 9000 FORMAT (/,/,' Warning: more than ',I7,2X,'iterations required to',
     +       ' obtain desired acuracy.')
      GO TO 80
   30 CONTINUE
      IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=9010)
 9010 FORMAT (' Error return from MA61B because',
     +    ' N is out of range.')
      IFLAG = -1
      GO TO 70
   40 CONTINUE
      IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=9020)
 9020 FORMAT (' Error return from MA61B because',
     +   ' NZ is out of range.')
      IFLAG = -2
      GO TO 70
   50 CONTINUE
      IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=9030)
 9030 FORMAT (' Error return from MA61B because',
     +    ' IAJ is out of range.')
      IFLAG = -4
      GO TO 70
   60 CONTINUE
      IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=9040)
 9040 FORMAT (' Error return from MA61B because',
     +    ' of error in previous call to MA61A.')
   70 CONTINUE
   80 CONTINUE
      INFO(1) = IFLAG
      RETURN
      END
      SUBROUTINE MA61F(N,NZ,A,INI,INJ,IAF,AF,DF,INJF,IK,B,R,E,F,G,KMAX,
     +                 EPS,LROW)
      REAL ZERO
      PARAMETER (ZERO=0.0E0)
      REAL EPS
      INTEGER IAF,KMAX,N,NZ,LROW
      REAL A(*),AF(*),B(*),DF(*),E(*),F(*),G(*),R(*)
      INTEGER IK(N,2),INI(*),INJ(*),INJF(*)
      REAL BB,D0,D1,L,R1
      INTEGER I,KITR
      EXTERNAL MA61G,MA61H
      INTRINSIC SQRT
      D0 = ZERO
      EPS = EPS**2
      DO 10 I = 1,N
         E(I) = B(I)
   10 CONTINUE
      CALL MA61G(N,AF,INJF,IAF,DF,IK,E,LROW)
      CALL MA61H(A,INI,INJ,NZ,N,E,R)
      R1 = ZERO
      DO 20 I = 1,N
         R(I) = R(I) - B(I)
         R1 = R1 + R(I)**2
         G(I) = R(I)
         B(I) = E(I)
   20 CONTINUE
      KITR = 0
      IF (R1.LT.EPS) GO TO 90
      CALL MA61G(N,AF,INJF,IAF,DF,IK,G,LROW)
      DO 30 I = 1,N
         E(I) = -G(I)
         D0 = D0 + R(I)*G(I)
   30 CONTINUE
   40 CONTINUE
      KITR = KITR + 1
      CALL MA61H(A,INI,INJ,NZ,N,E,F)
      L = ZERO
      DO 50 I = 1,N
         L = L + E(I)*F(I)
   50 CONTINUE
      L = D0/L
      R1 = ZERO
      DO 60 I = 1,N
         B(I) = B(I) + L*E(I)
         R(I) = R(I) + L*F(I)
         R1 = R1 + R(I)*R(I)
         G(I) = R(I)
   60 CONTINUE
      IF (R1.LE.EPS .OR. KITR.GE.KMAX) GO TO 90
      CALL MA61G(N,AF,INJF,IAF,DF,IK,G,LROW)
      D1 = ZERO
      DO 70 I = 1,N
         D1 = R(I)*G(I) + D1
   70 CONTINUE
      BB = D1/D0
      D0 = D1
      DO 80 I = 1,N
         E(I) = -G(I) + BB*E(I)
   80 CONTINUE
      GO TO 40
   90 CONTINUE
      KMAX = KITR
      EPS = SQRT(R1)
      RETURN
      END
      SUBROUTINE MA61H(A,INI,INJ,NZ,N,B,Z)
      INTEGER N,NZ
      REAL A(*),B(*),Z(*)
      INTEGER INI(*),INJ(*)
      INTEGER I,J,K
      DO 10 I = 1,N
         Z(I) = B(I)
   10 CONTINUE
      IF (NZ.LE.0) GO TO 30
      DO 20 K = 1,NZ
         I = INI(K)
         J = INJ(K)
         Z(I) = Z(I) + A(K)*B(J)
         Z(J) = Z(J) + A(K)*B(I)
   20 CONTINUE
   30 CONTINUE
      RETURN
      END
