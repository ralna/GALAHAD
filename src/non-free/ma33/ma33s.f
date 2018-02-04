* *******************************************************************
* COPYRIGHT (c) 1977 Hyprotech UK, 2006 CCLRC
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
*######DATE 16 Mar 2006

C  Thread-safe version of MA30

C IF  THE USER REQUIRES A MORE CONVENIENT DATA INTERFACE THEN THE MA28
C     PACKAGE SHOULD BE USED.  THE MA28 SUBROUTINES CALL THE MA33
C     SUBROUTINES AFTER CHECKING THE USER'S INPUT DATA AND OPTIONALLY
C     USING MC23A/AD TO PERMUTE THE MATRIX TO BLOCK TRIANGULAR FORM.
C THIS PACKAGE OF SUBROUTINES (MA33A/AD, MA33B/BD, MA33C/CD AND
C     MA33D/DD) PERFORMS OPERATIONS PERTINENT TO THE SOLUTION OF A
C     GENERAL SPARSE N BY N SYSTEM OF LINEAR EQUATIONS (I.E. SOLVE
C     AX=B). STRUCTUALLY SINGULAR MATRICES ARE PERMITTED INCLUDING
C     THOSE WITH ROW OR COLUMNS CONSISTING ENTIRELY OF ZEROS (I.E.
C     INCLUDING RECTANGULAR MATRICES).  IT IS ASSUMED THAT THE
C     NON-ZEROS OF THE MATRIX A DO NOT DIFFER WIDELY IN SIZE.  IF
C     NECESSARY A PRIOR CALL OF THE SCALING SUBROUTINE MC19A/AD MAY BE
C     MADE.
C A DISCUSSION OF THE DESIGN OF THESE SUBROUTINES IS GIVEN BY DUFF AND
C     REID (ACM TRANS MATH SOFTWARE 5 PP 18-35,1979 (CSS 48)) WHILE
C     FULLER DETAILS OF THE IMPLEMENTATION ARE GIVEN IN DUFF (HARWELL
C     REPORT AERE-R 8730,1977).  THE ADDITIONAL PIVOTING OPTION IN
C     MA33A/AD AND THE USE OF DROP TOLERANCES WERE ADDED TO THE PACKAGE
C     AFTER JOINT WORK WITH REID, SCHAUMBURG, WASNIEWSKI AND ZLATEV
C     (DUFF, REID, SCHAUMBURG, WASNIEWSKI AND ZLATEV, HARWELL REPORT
C     CSS 135, 1983).
C
C Thread-safe addaptation by Nick Gould, March 2006
C
C  Modified for GALAHAD: INTENT added for all dummy arguments
C
      SUBROUTINE MA33I(ICNTL,CNTL)
      INTEGER ICNTL(10)
      REAL CNTL(5)
      ICNTL(1) = 6
      ICNTL(2) = 1
      ICNTL(3) = 1
      ICNTL(4) = 0
      ICNTL(5) = 32768
      ICNTL(6) = 0
      ICNTL(7) = 0
      ICNTL(8) = 0
      ICNTL(9) = 0
      ICNTL(10) = 0
      CNTL(1) = 0.0001
      CNTL(2) = 0.0
      CNTL(3) = 0
      CNTL(4) = 0
      CNTL(5) = 0
      RETURN
      END
      SUBROUTINE MA33A(NN,ICN,A,LICN,LENR,LENRL,IDISP,IP,IQ,IRN,LIRN,
     +                  LENC,IFIRST,LASTR,NEXTR,LASTC,NEXTC,IPTR,IPC,U,
     +                  IFLAG,ICNTL,CNTL,INFO,RINFO)
      INTEGER, INTENT( IN ) :: nn, licn, lirn
      INTEGER, INTENT( OUT ) :: iflag
      REAL, INTENT( INOUT ) :: u
      INTEGER, INTENT( INOUT ), DIMENSION( licn ) ::  ICN
      INTEGER, INTENT( INOUT ), DIMENSION( nn ) ::  LENR
      INTEGER, INTENT( OUT ), DIMENSION( nn ) ::  LENRL
      INTEGER, INTENT( INOUT ), DIMENSION( 2 ) ::  IDISP
      INTEGER, INTENT( INOUT ), DIMENSION( nn ) ::  IP, IQ
      INTEGER, INTENT( OUT ), DIMENSION( lirn ) :: IRN
      INTEGER, INTENT( OUT ), DIMENSION( nn ) ::  IPC, IPTR
      INTEGER, INTENT( OUT ), DIMENSION( nn ) ::  LENC, IFIRST
      INTEGER, INTENT( OUT ), DIMENSION( nn ) ::  LASTR, NEXTR
      INTEGER, INTENT( OUT ), DIMENSION( nn ) ::  LASTC, NEXTC
      INTEGER, INTENT( OUT ), DIMENSION( 10 ) :: INFO
      INTEGER, INTENT( IN ), DIMENSION( 10 ) :: ICNTL
      REAL, INTENT( INOUT ), DIMENSION( licn ) :: A
      REAL, INTENT( OUT ), DIMENSION( 5 ) :: RINFO
      REAL, INTENT( IN ), DIMENSION( 5 ) :: CNTL
      REAL ZERO,UMAX
      PARAMETER (ZERO=0.0E0,UMAX=0.9999E0)
      REAL AANEW,AMAX,ANEW,AU,PIVR,PIVRAT,SCALE
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )
      INTEGER ( KIND = long ) :: N_long, NZ_long, JCOST_long, KCOST_long
      INTEGER COLUPD,DISPC,I,I1,I2,IACTIV,IBEG,IDISPC,IDROP,IDUMMY,IEND,
     +        IFILL,IFIR,II,III,IJFIR,IJP1,IJPOS,ILAST,INDROW,IOP,IPIV,
     +        IPOS,IROWS,ISING,ISRCH,ISTART,ISW,ISW1,ITOP,J,J1,J2,JBEG,
     +        JCOUNT,JDIFF,JDUMMY,JEND,JJ,JMORE,JNEW,JNPOS,JOLD,
     +        JPIV,JPOS,JROOM,JVAL,JZER,JZERO,K,KDROP,L,LC,LENPIV,
     +        LL,LR,MOREI,MSRCH,N,NBLOCK,NC,NNM1,NR,NUM,NZ,NZ2,NZCOL,
     +        NZMIN,NZPC,NZROW,OLDEND,OLDPIV,PIVEND,PIVOT,PIVROW,ROWI
      EXTERNAL MA33DD
      INFO(1) = 0
      INFO(2) = 0
      INFO(3) = NN
      INFO(4) = 0
      INFO(5) = IDISP(1) - 1
      INFO(6) = 0
      INFO(7) = 0
      INFO(8) = 0
      INFO(9) = 0
      INFO(10) = 0
      RINFO(1) = CNTL(1)
      RINFO(2) = 0.0
      RINFO(3) = 0
      RINFO(4) = 0
      RINFO(5) = 0
      MSRCH = ICNTL(5)
      MOREI = 0
      IFLAG = 0
      U = MIN(U,UMAX)
      U = MAX(U,ZERO)
      IBEG = IDISP(1)
      IACTIV = IDISP(2)
      NZROW = LICN - IACTIV + 1
      INFO(5) = NZROW + INFO(5)
      NUM = 1
      IPTR(1) = IACTIV
      IF (NN.EQ.1) GO TO 20
      NNM1 = NN - 1
      DO 10 I = 1,NNM1
        IF (IP(I).LT.0) NUM = NUM + 1
        IPTR(I+1) = IPTR(I) + LENR(I)
   10 CONTINUE
   20 CONTINUE
      ILAST = 0
      DO 1000 NBLOCK = 1,NUM
        ISTART = ILAST + 1
        DO 30 IROWS = ISTART,NN
          IF (IP(IROWS).LT.0) GO TO 40
   30   CONTINUE
        IROWS = NN
   40   CONTINUE
        ILAST = IROWS
        N = ILAST - ISTART + 1
        IF (N.NE.1) GO TO 90
        LENRL(ILAST) = 0
        ISING = ISTART
        IF (LENR(ILAST).NE.0) GO TO 50
        INFO(3) = INFO(3) - 1
        ISING = -ISING
        IF (IFLAG.NE.2 .AND. IFLAG.NE.-5) IFLAG = 1
        IF (ICNTL(2).EQ.0) GO TO 80
        IDISP(2) = IACTIV
        IFLAG = -1
        IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=99999)
        GO TO 1120
   50   CONTINUE
        SCALE = ABS(A(IACTIV))
        IF (SCALE.EQ.ZERO) GO TO 60
        IF (ICNTL(6).NE.0) RINFO(2) = MAX(RINFO(2),SCALE)
        GO TO 70
   60   CONTINUE
        ISING = -ISING
        INFO(3) = INFO(3) - 1
        IPTR(ILAST) = 0
        IF (IFLAG.NE.-5) IFLAG = 2
        IF (ICNTL(3).EQ.0) GO TO 70
        IDISP(2) = IACTIV
        IFLAG = -2
        IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=99998)
        GO TO 1120
   70   CONTINUE
        A(IBEG) = A(IACTIV)
        ICN(IBEG) = ICN(IACTIV)
        IACTIV = IACTIV + 1
        IPTR(ISTART) = 0
        IBEG = IBEG + 1
        NZROW = NZROW - 1
   80   CONTINUE
        LASTR(ISTART) = ISTART
        IPC(ISTART) = -ISING
        GO TO 1000
   90   CONTINUE
        ITOP = LICN
        IF (ILAST.NE.NN) ITOP = IPTR(ILAST+1) - 1
        DO 100 I = ISTART,ILAST
          LENRL(I) = 0
          LENC(I) = 0
  100   CONTINUE
        IF (ITOP-IACTIV.LT.LIRN) GO TO 110
        INFO(4) = ITOP - IACTIV + 1
        PIVOT = ISTART - 1
        GO TO 1100
  110   CONTINUE
        DO 120 II = IACTIV,ITOP
          I = ICN(II)
          LENC(I) = LENC(I) + 1
  120   CONTINUE
        IPC(ILAST) = LIRN + 1
        J1 = ISTART + 1
        DO 130 JJ = J1,ILAST
          J = ILAST - JJ + J1 - 1
          IPC(J) = IPC(J+1) - LENC(J+1)
  130   CONTINUE
        DO 150 INDROW = ISTART,ILAST
          J1 = IPTR(INDROW)
          J2 = J1 + LENR(INDROW) - 1
          IF (J1.GT.J2) GO TO 150
          DO 140 JJ = J1,J2
            J = ICN(JJ)
            IPOS = IPC(J) - 1
            IRN(IPOS) = INDROW
            IPC(J) = IPOS
  140     CONTINUE
  150   CONTINUE
        DISPC = IPC(ISTART)
        NZCOL = LIRN - DISPC + 1
        INFO(4) = MAX(NZCOL,INFO(4))
        NZMIN = 1
        DO 160 I = 1,N
          IFIRST(I) = 0
  160   CONTINUE
        DO 180 JJ = ISTART,ILAST
          J = ILAST - JJ + ISTART
          NZ = LENC(J)
          IF (NZ.NE.0) GO TO 170
          IPC(J) = 0
          GO TO 180
  170     CONTINUE
          IF (ICNTL(5).LE.NN) GO TO 180
          ISW = IFIRST(NZ)
          IFIRST(NZ) = -J
          LASTC(J) = 0
          NEXTC(J) = -ISW
          ISW1 = ABS(ISW)
          IF (ISW.NE.0) LASTC(ISW1) = J
  180   CONTINUE
        DO 210 II = ISTART,ILAST
          I = ILAST - II + ISTART
          NZ = LENR(I)
          IF (NZ.NE.0) GO TO 190
          IPTR(I) = 0
          LASTR(I) = 0
          GO TO 210
  190     CONTINUE
          ISW = IFIRST(NZ)
          IFIRST(NZ) = I
          IF (ISW.GT.0) GO TO 200
          NEXTR(I) = 0
          LASTR(I) = ISW
          GO TO 210
  200     CONTINUE
          NEXTR(I) = ISW
          LASTR(I) = LASTR(ISW)
          LASTR(ISW) = I
  210   CONTINUE
        N_long = INT(N,long)
        DO 980 PIVOT = ISTART,ILAST
          NZ2 = NZMIN
          JCOST_long = N_long*N_long
          DO 340 L = 1,2
            PIVRAT = ZERO
            ISRCH = 1
            LL = L
            DO 330 NZ = NZ2,N
              NZ_long = INT(NZ,long)
              IF (JCOST_long.LE. (NZ_long-1)**2) GO TO 420
              IJFIR = IFIRST(NZ)
              IF (IJFIR.LT.0) THEN
                LL = 2
                IJFIR = -IJFIR
                GO TO 290
              ELSE IF (IJFIR.EQ.0) THEN
                IF (LL.EQ.1) NZMIN = NZ + 1
                GO TO 330
              END IF
              LL = 2
              DO 270 IDUMMY = 1,N
                IF (JCOST_long.LE. (NZ_long-1)**2) GO TO 420
                IF (ISRCH.GT.MSRCH) GO TO 420
                IF (IJFIR.EQ.0) GO TO 280
                I = IJFIR
                IJFIR = NEXTR(I)
                AMAX = ZERO
                J1 = IPTR(I) + LENRL(I)
                J2 = IPTR(I) + LENR(I) - 1
                DO 250 JJ = J1,J2
                  AMAX = MAX(AMAX,ABS(A(JJ)))
  250           CONTINUE
                AU = AMAX*U
                ISRCH = ISRCH + 1
                DO 260 JJ = J1,J2
                  IF (ABS(A(JJ)).LE.AU .AND. L.EQ.1) GO TO 260
                  J = ICN(JJ)
                  KCOST_long = (NZ_long-1)* (INT(LENC(J),long)-1)
                  IF (KCOST_long.GT.JCOST_long) GO TO 260
                  PIVR = ZERO
                  IF (AMAX.NE.ZERO) PIVR = ABS(A(JJ))/AMAX
                  IF (KCOST_long.EQ.JCOST_long .AND. (PIVR.LE.PIVRAT
     +                .OR.ICNTL(5).GT.NN+1)) GO TO 260
                  JCOST_long = KCOST_long
                  IJPOS = JJ
                  IPIV = I
                  JPIV = J
                  IF (MSRCH.GT.NN+1 .AND. 
     +              JCOST_long.LE.(NZ_long-1)**2) GO TO 420
                  PIVRAT = PIVR
  260           CONTINUE
  270         CONTINUE
  280         CONTINUE
              IJFIR = IFIRST(NZ)
              IJFIR = -LASTR(IJFIR)
  290         CONTINUE
              IF (JCOST_long.LE.NZ_long*(NZ_long-1)) GO TO 420
              IF (MSRCH.LE.NN) GO TO 330
              DO 320 IDUMMY = 1,N
                IF (IJFIR.EQ.0) GO TO 330
                J = IJFIR
                IJFIR = NEXTC(IJFIR)
                I1 = IPC(J)
                I2 = I1 + NZ - 1
                DO 310 II = I1,I2
                  I = IRN(II)
                  KCOST_long = 
     +              (NZ_long-1) * (INT(LENR(I)-LENRL(I),long)-1)
                  IF (KCOST_long.GE.JCOST_long) GO TO 310
                  J1 = IPTR(I) + LENRL(I)
                  J2 = IPTR(I) + LENR(I) - 1
                  AMAX = ZERO
                  DO 300 JJ = J1,J2
                    AMAX = MAX(AMAX,ABS(A(JJ)))
                    IF (ICN(JJ).EQ.J) JPOS = JJ
  300             CONTINUE
                  IF (ABS(A(JPOS)).LE.AMAX*U .AND. L.EQ.1) GO TO 310
                  JCOST_long = KCOST_long
                  IPIV = I
                  JPIV = J
                  IJPOS = JPOS
                  IF (AMAX.NE.ZERO) PIVRAT = ABS(A(JPOS))/AMAX
                  IF (JCOST_long.LE.NZ_long * (NZ_long-1)) GO TO 420
  310           CONTINUE
  320         CONTINUE
  330       CONTINUE
            MSRCH = N
            INFO(3) = INFO(3) - 1
  340     CONTINUE
          IF (IFLAG.NE.2 .AND. IFLAG.NE.-5) IFLAG = 1
          INFO(3) = INFO(3) - ILAST + PIVOT + 1
          IF (ICNTL(2).EQ.0) GO TO 350
          IDISP(2) = IACTIV
          IFLAG = -1
          IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=99999)
          GO TO 1120
  350     CONTINUE
          K = PIVOT - 1
          DO 390 I = ISTART,ILAST
            IF (LASTR(I).NE.0) GO TO 390
            K = K + 1
            LASTR(I) = K
            IF (LENRL(I).EQ.0) GO TO 380
            INFO(5) = MAX(INFO(5),NZROW+IBEG-1+MOREI+LENRL(I))
            IF (IACTIV-IBEG.GE.LENRL(I)) GO TO 360
            CALL MA33D(A,ICN,IPTR(ISTART),N,IACTIV,ITOP,.TRUE.,INFO)
            IF (IACTIV-IBEG.GE.LENRL(I)) GO TO 360
            MOREI = MOREI + IBEG - IDISP(1)
            IBEG = IDISP(1)
            IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=99997)
            IFLAG = -5
            IF (ICNTL(4).NE.0) GO TO 1090
  360       CONTINUE
            J1 = IPTR(I)
            J2 = J1 + LENRL(I) - 1
            IPTR(I) = 0
            DO 370 JJ = J1,J2
              A(IBEG) = A(JJ)
              ICN(IBEG) = ICN(JJ)
              ICN(JJ) = 0
              IBEG = IBEG + 1
  370       CONTINUE
            NZROW = NZROW - LENRL(I)
  380       CONTINUE
            IF (K.EQ.ILAST) GO TO 400
  390     CONTINUE
  400     CONTINUE
          K = PIVOT - 1
          DO 410 I = ISTART,ILAST
            IF (IPC(I).NE.0) GO TO 410
            K = K + 1
            IPC(I) = K
            IF (K.EQ.ILAST) GO TO 990
  410     CONTINUE
  420     CONTINUE
          ISING = PIVOT
          IF (A(IJPOS).NE.ZERO) GO TO 430
          ISING = -ISING
          IF (IFLAG.NE.-5) IFLAG = 2
          IF (ICNTL(3).EQ.0) GO TO 430
          IDISP(2) = IACTIV
          IFLAG = -2
          IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=99998)
          GO TO 1120
  430     CONTINUE
          OLDPIV = IPTR(IPIV) + LENRL(IPIV)
          OLDEND = IPTR(IPIV) + LENR(IPIV) - 1
          IF (ICNTL(5).LE.NN) GO TO 460
          COLUPD = NN + 1
          DO 450 JJ = OLDPIV,OLDEND
            J = ICN(JJ)
            LC = LASTC(J)
            NC = NEXTC(J)
            NEXTC(J) = -COLUPD
            IF (JJ.NE.IJPOS) COLUPD = J
            IF (NC.NE.0) LASTC(NC) = LC
            IF (LC.EQ.0) GO TO 440
            NEXTC(LC) = NC
            GO TO 450
  440       CONTINUE
            NZ = LENC(J)
            ISW = IFIRST(NZ)
            IF (ISW.GT.0) LASTR(ISW) = -NC
            IF (ISW.LT.0) IFIRST(NZ) = -NC
  450     CONTINUE
  460     CONTINUE
          I1 = IPC(JPIV)
          I2 = I1 + LENC(JPIV) - 1
          DO 480 II = I1,I2
            I = IRN(II)
            LR = LASTR(I)
            NR = NEXTR(I)
            IF (NR.NE.0) LASTR(NR) = LR
            IF (LR.LE.0) GO TO 470
            NEXTR(LR) = NR
            GO TO 480
  470       CONTINUE
            NZ = LENR(I) - LENRL(I)
            IF (NR.NE.0) IFIRST(NZ) = NR
            IF (NR.EQ.0) IFIRST(NZ) = LR
  480     CONTINUE
          IF (OLDPIV.EQ.IJPOS) GO TO 490
          AU = A(OLDPIV)
          A(OLDPIV) = A(IJPOS)
          A(IJPOS) = AU
          ICN(IJPOS) = ICN(OLDPIV)
          ICN(OLDPIV) = JPIV
  490     CONTINUE
          INFO(5) = MAX(INFO(5),NZROW+IBEG-1+MOREI+LENR(IPIV))
          IF (IACTIV-IBEG.GE.LENR(IPIV)) GO TO 500
          CALL MA33D(A,ICN,IPTR(ISTART),N,IACTIV,ITOP,.TRUE.,INFO)
          OLDPIV = IPTR(IPIV) + LENRL(IPIV)
          OLDEND = IPTR(IPIV) + LENR(IPIV) - 1
          IF (IACTIV-IBEG.GE.LENR(IPIV)) GO TO 500
          MOREI = MOREI + IBEG - IDISP(1)
          IBEG = IDISP(1)
          IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=99997)
          IFLAG = -5
          IF (ICNTL(4).NE.0) GO TO 1090
          IF (IACTIV-IBEG.GE.LENR(IPIV)) GO TO 500
          IFLAG = -4
          GO TO 1090
  500     CONTINUE
          IJPOS = 0
          J1 = IPTR(IPIV)
          DO 530 JJ = J1,OLDEND
            A(IBEG) = A(JJ)
            ICN(IBEG) = ICN(JJ)
            IF (IJPOS.NE.0) GO TO 510
            IF (ICN(JJ).EQ.JPIV) IJPOS = IBEG
            ICN(JJ) = 0
            GO TO 520
  510       CONTINUE
            K = IBEG - IJPOS
            J = ICN(JJ)
            ICN(JJ) = IQ(J)
            IQ(J) = -K
  520       CONTINUE
            IBEG = IBEG + 1
  530     CONTINUE
          IJP1 = IJPOS + 1
          PIVEND = IBEG - 1
          LENPIV = PIVEND - IJPOS
          NZROW = NZROW - LENRL(IPIV) - 1
          IPTR(IPIV) = OLDPIV + 1
          IF (LENPIV.EQ.0) IPTR(IPIV) = 0
          DO 560 JJ = IJPOS,PIVEND
            J = ICN(JJ)
            I1 = IPC(J)
            LENC(J) = LENC(J) - 1
            I2 = IPC(J) + LENC(J) - 1
            IF (I2.LT.I1) GO TO 550
            DO 540 II = I1,I2
              IF (IRN(II).NE.IPIV) GO TO 540
              IRN(II) = IRN(I2+1)
              GO TO 550
  540       CONTINUE
  550       CONTINUE
            IRN(I2+1) = 0
  560     CONTINUE
          NZCOL = NZCOL - LENPIV - 1
          NZPC = LENC(JPIV)
          IF (NZPC.EQ.0) GO TO 900
          DO 840 III = 1,NZPC
            II = IPC(JPIV) + III - 1
            I = IRN(II)
            IDROP = 0
            J1 = IPTR(I) + LENRL(I)
            IEND = IPTR(I) + LENR(I) - 1
            DO 570 JJ = J1,IEND
              IF (ICN(JJ).NE.JPIV) GO TO 570
              AU = ZERO
              IF (A(IJPOS).NE.ZERO) AU = -A(JJ)/A(IJPOS)
              IF (ICNTL(6).NE.0) RINFO(2) = MAX(RINFO(2),ABS(AU))
              A(JJ) = A(J1)
              A(J1) = AU
              ICN(JJ) = ICN(J1)
              ICN(J1) = JPIV
              LENRL(I) = LENRL(I) + 1
              GO TO 580
  570       CONTINUE
  580       CONTINUE
            IF (LENPIV.EQ.0) GO TO 840
            ROWI = J1 + 1
            IOP = 0
            IF (ROWI.GT.IEND) GO TO 650
            DO 590 JJ = ROWI,IEND
              J = ICN(JJ)
              IF (IQ(J).GT.0) GO TO 590
              IOP = IOP + 1
              PIVROW = IJPOS - IQ(J)
              A(JJ) = A(JJ) + AU*A(PIVROW)
              IF (ICNTL(6).NE.0) RINFO(2) = MAX(ABS(A(JJ)),RINFO(2))
              ICN(PIVROW) = -ICN(PIVROW)
              IF (ABS(A(JJ)).LT.CNTL(2)) IDROP = IDROP + 1
  590       CONTINUE
            IF (IDROP.EQ.0) GO TO 650
            JNEW = ROWI
            DO 630 JJ = ROWI,IEND
              IF (ABS(A(JJ)).LT.CNTL(2)) GO TO 600
              A(JNEW) = A(JJ)
              ICN(JNEW) = ICN(JJ)
              JNEW = JNEW + 1
              GO TO 630
  600         CONTINUE
              J = ICN(JJ)
              I1 = IPC(J)
              I2 = I1 + LENC(J) - 1
              DO 610 II = I1,I2
                IF (IRN(II).EQ.I) GO TO 620
  610         CONTINUE
  620         CONTINUE
              IRN(II) = IRN(I2)
              IRN(I2) = 0
              LENC(J) = LENC(J) - 1
              IF (ICNTL(5).LE.NN) GO TO 630
              IF (NEXTC(J).LT.0) GO TO 630
              LC = LASTC(J)
              NC = NEXTC(J)
              NEXTC(J) = -COLUPD
              COLUPD = J
              IF (NC.NE.0) LASTC(NC) = LC
              IF (LC.EQ.0) GO TO 622
              NEXTC(LC) = NC
              GO TO 630
  622         CONTINUE
              NZ = LENC(J) + 1
              ISW = IFIRST(NZ)
              IF (ISW.GT.0) LASTR(ISW) = -NC
              IF (ISW.LT.0) IFIRST(NZ) = -NC
  630       CONTINUE
            DO 640 JJ = JNEW,IEND
              ICN(JJ) = 0
  640       CONTINUE
            IDROP = IEND + 1 - JNEW
            IEND = JNEW - 1
            LENR(I) = LENR(I) - IDROP
            NZROW = NZROW - IDROP
            NZCOL = NZCOL - IDROP
            INFO(6) = INFO(6) + IDROP
  650       CONTINUE
            IFILL = LENPIV - IOP
            IF (IFILL.EQ.0) GO TO 750
            INFO(5) = MAX(INFO(5),MOREI+IBEG-1+NZROW+IFILL+LENR(I))
            DO 660 JDIFF = 1,IFILL
              JNPOS = IEND + JDIFF
              IF (JNPOS.GT.LICN) GO TO 670
              IF (ICN(JNPOS).NE.0) GO TO 670
  660       CONTINUE
            IEND = IEND + 1
            GO TO 750
  670       CONTINUE
            JMORE = IFILL - JDIFF + 1
            I1 = IPTR(I)
            DO 680 JDIFF = 1,JMORE
              JNPOS = I1 - JDIFF
              IF (JNPOS.LT.IACTIV) GO TO 690
              IF (ICN(JNPOS).NE.0) GO TO 700
  680       CONTINUE
  690       CONTINUE
            JNPOS = I1 - JMORE
            GO TO 710
  700       CONTINUE
            JNPOS = IACTIV - LENR(I) - IFILL
  710       CONTINUE
            IF (JNPOS.GE.IBEG) GO TO 730
            CALL MA33D(A,ICN,IPTR(ISTART),N,IACTIV,ITOP,.TRUE.,INFO)
            I1 = IPTR(I)
            IEND = I1 + LENR(I) - 1
            JNPOS = IACTIV - LENR(I) - IFILL
            IF (JNPOS.GE.IBEG) GO TO 730
            MOREI = MOREI + IBEG - IDISP(1) - LENPIV - 1
            IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=99997)
            IFLAG = -5
            IF (ICNTL(4).NE.0) GO TO 1090
            IBEG = IDISP(1)
            ICN(IBEG) = JPIV
            A(IBEG) = A(IJPOS)
            IJPOS = IBEG
            DO 720 JJ = IJP1,PIVEND
              IBEG = IBEG + 1
              A(IBEG) = A(JJ)
              ICN(IBEG) = ICN(JJ)
  720       CONTINUE
            IJP1 = IJPOS + 1
            PIVEND = IBEG
            IBEG = IBEG + 1
            IF (JNPOS.GE.IBEG) GO TO 730
            IFLAG = -4
            GO TO 1090
  730       CONTINUE
            IACTIV = MIN(IACTIV,JNPOS)
            IPTR(I) = JNPOS
            DO 740 JJ = I1,IEND
              A(JNPOS) = A(JJ)
              ICN(JNPOS) = ICN(JJ)
              JNPOS = JNPOS + 1
              ICN(JJ) = 0
  740       CONTINUE
            IEND = JNPOS
  750       CONTINUE
            NZROW = NZROW + IFILL
            IDROP = 0
            DO 830 JJ = IJP1,PIVEND
              J = ICN(JJ)
              IF (J.LT.0) GO TO 820
              ANEW = AU*A(JJ)
              AANEW = ABS(ANEW)
              IF (AANEW.GE.CNTL(2)) GO TO 760
              IDROP = IDROP + 1
              INFO(6) = INFO(6) + 1
              NZROW = NZROW - 1
              INFO(5) = INFO(5) - 1
              IFILL = IFILL - 1
              GO TO 830
  760         CONTINUE
              IF (ICNTL(6).NE.0) RINFO(2) = MAX(AANEW,RINFO(2))
              A(IEND) = ANEW
              ICN(IEND) = J
              IEND = IEND + 1
              INFO(4) = MAX(INFO(4),NZCOL+LENC(J)+1)
              JEND = IPC(J) + LENC(J)
              JROOM = NZPC - III + 1 + LENC(J)
              IF (JEND.GT.LIRN) GO TO 770
              IF (IRN(JEND).EQ.0) GO TO 810
  770         CONTINUE
              IF (JROOM.LT.DISPC) GO TO 780
              CALL MA33D(A,IRN,IPC(ISTART),N,DISPC,LIRN,.FALSE.,INFO)
              IF (JROOM.LT.DISPC) GO TO 780
              JROOM = DISPC - 1
              IF (JROOM.GE.LENC(J)+1) GO TO 780
              GO TO 1100
  780         CONTINUE
              JBEG = IPC(J)
              JEND = IPC(J) + LENC(J) - 1
              JZERO = DISPC - 1
              DISPC = DISPC - JROOM
              IDISPC = DISPC
              DO 790 II = JBEG,JEND
                IRN(IDISPC) = IRN(II)
                IRN(II) = 0
                IDISPC = IDISPC + 1
  790         CONTINUE
              IPC(J) = DISPC
              JEND = IDISPC
              DO 800 II = JEND,JZERO
                IRN(II) = 0
  800         CONTINUE
  810         CONTINUE
              IRN(JEND) = I
              NZCOL = NZCOL + 1
              LENC(J) = LENC(J) + 1
              GO TO 830
  820         CONTINUE
              ICN(JJ) = -J
  830       CONTINUE
            IF (IDROP.EQ.0) GO TO 834
            DO 832 KDROP = 1,IDROP
              ICN(IEND) = 0
              IEND = IEND + 1
  832       CONTINUE
  834       CONTINUE
            LENR(I) = LENR(I) + IFILL
  840     CONTINUE
          I1 = IPC(JPIV)
          I2 = IPC(JPIV) + LENC(JPIV) - 1
          NZCOL = NZCOL - LENC(JPIV)
          DO 890 II = I1,I2
            I = IRN(II)
            IRN(II) = 0
            NZ = LENR(I) - LENRL(I)
            IF (NZ.NE.0) GO TO 850
            LASTR(I) = 0
            GO TO 890
  850       CONTINUE
            IFIR = IFIRST(NZ)
            IFIRST(NZ) = I
            IF (IFIR.LT.0) THEN
              LASTR(I) = IFIR
              NEXTR(I) = 0
            ELSE IF (IFIR.EQ.0) THEN
              LASTR(I) = 0
              NEXTR(I) = 0
              NZMIN = MIN(NZMIN,NZ)
            ELSE
              LASTR(I) = LASTR(IFIR)
              NEXTR(I) = IFIR
              LASTR(IFIR) = I
            END IF
  890     CONTINUE
  900     CONTINUE
          IPC(JPIV) = -ISING
          LASTR(IPIV) = PIVOT
          IF (LENPIV.EQ.0) GO TO 980
          NZROW = NZROW - LENPIV
          JVAL = IJP1
          JZER = IPTR(IPIV)
          IPTR(IPIV) = 0
          DO 910 JCOUNT = 1,LENPIV
            J = ICN(JVAL)
            IQ(J) = ICN(JZER)
            ICN(JZER) = 0
            JVAL = JVAL + 1
            JZER = JZER + 1
  910     CONTINUE
          IF (ICNTL(5).GT.NN) GO TO 920
          DO 916 JJ = IJP1,PIVEND
            J = ICN(JJ)
            NZ = LENC(J)
            IF (NZ.NE.0) GO TO 914
            IPC(J) = 0
            GO TO 916
  914       CONTINUE
            NZMIN = MIN(NZMIN,NZ)
  916     CONTINUE
          GO TO 980
  920     CONTINUE
          JJ = COLUPD
          DO 970 JDUMMY = 1,NN
            J = JJ
            IF (J.EQ.NN+1) GO TO 980
            JJ = -NEXTC(J)
            NZ = LENC(J)
            IF (NZ.NE.0) GO TO 924
            IPC(J) = 0
            GO TO 970
  924       CONTINUE
            IFIR = IFIRST(NZ)
            LASTC(J) = 0
            IF (IFIR.LT.0) THEN
              IFIRST(NZ) = -J
              IFIR = -IFIR
              LASTC(IFIR) = J
              NEXTC(J) = IFIR
              GO TO 970
            ELSE IF (IFIR.EQ.0) THEN
              IFIRST(NZ) = -J
              NEXTC(J) = 0
            ELSE
              LC = -LASTR(IFIR)
              LASTR(IFIR) = -J
              NEXTC(J) = LC
              IF (LC.NE.0) LASTC(LC) = J
            END IF
            NZMIN = MIN(NZMIN,NZ)
  970     CONTINUE
  980   CONTINUE
  990   CONTINUE
        IF (ILAST.NE.NN) IACTIV = IPTR(ILAST+1)
 1000 CONTINUE
      IF (INFO(3).EQ.NN) GO TO 1020
      DO 1010 I = 1,NN
        IF (IPC(I).LT.0) GO TO 1010
        ISING = IPC(I)
        IQ(ISING) = -IQ(ISING)
        IPC(I) = -ISING
 1010 CONTINUE
 1020 CONTINUE
      ISTART = IDISP(1)
      IEND = IBEG - 1
      IF (IEND.LT.ISTART) GO TO 1040
      DO 1030 JJ = ISTART,IEND
        JOLD = ICN(JJ)
        ICN(JJ) = -IPC(JOLD)
 1030 CONTINUE
 1040 DO 1050 II = 1,NN
        I = LASTR(II)
        NEXTR(I) = LENR(II)
        IPTR(I) = LENRL(II)
 1050 CONTINUE
      DO 1060 I = 1,NN
        LENRL(I) = IPTR(I)
        LENR(I) = NEXTR(I)
 1060 CONTINUE
      DO 1070 II = 1,NN
        I = LASTR(II)
        J = -IPC(II)
        NEXTR(I) = ABS(IP(II)+0)
        IPTR(J) = ABS(IQ(II)+0)
 1070 CONTINUE
      DO 1080 I = 1,NN
        IF (IP(I).LT.0) NEXTR(I) = -NEXTR(I)
        IP(I) = NEXTR(I)
        IF (IQ(I).LT.0) IPTR(I) = -IPTR(I)
        IQ(I) = IPTR(I)
 1080 CONTINUE
      IP(NN) = ABS(IP(NN)+0)
      IDISP(2) = IEND
      GO TO 1120
 1090 CONTINUE
      IDISP(2) = IACTIV
      IF (ICNTL(1).LE.0) GO TO 1120
      WRITE (ICNTL(1),FMT=99996)
      GO TO 1110
 1100 CONTINUE
      IF (IFLAG.EQ.-5) IFLAG = -6
      IF (IFLAG.NE.-6) IFLAG = -3
      IDISP(2) = IACTIV
      IF (ICNTL(1).LE.0) GO TO 1120
      IF (IFLAG.EQ.-3) WRITE (ICNTL(1),FMT=99995)
      IF (IFLAG.EQ.-6) WRITE (ICNTL(1),FMT=99994)
 1110 CONTINUE
      PIVOT = PIVOT - ISTART + 1
      WRITE (ICNTL(1),FMT=99993) PIVOT,NBLOCK,ISTART,ILAST
      IF (PIVOT.EQ.0) WRITE (ICNTL(1),FMT=99992) INFO(4)
 1120 RETURN
99999 FORMAT (' Error return from MA33A/AD because matrix is',
     +        ' structurally singular' )
99998 FORMAT (' Error return from MA33A/AD because matrix is',
     +        ' numerically singular')
99997 FORMAT (' LU decomposition destroyed to create more space')
99996 FORMAT (' Error return from MA33A/AD because LICN not big',
     +        ' enough')
99995 FORMAT (' Error return from MA33A/AD because LIRN not big',
     +        ' enough')
99994 FORMAT (' Error return from MA33A/AD LIRN and LICN too small')
99993 FORMAT (' At stage ',I5,' in block ',I5,' with first row ',I5,
     +        ' and last row ',I5)
99992 FORMAT (' To continue set LIRN to at least ',I8)
      END
      SUBROUTINE MA33B(N,ICN,A,LICN,LENR,LENRL,IDISP,IP,IQ,W,IW,IFLAG,
     *                  ICNTL,CNTL,RINFO)
      REAL ZERO,ONE
      PARAMETER (ZERO=0.0E0,ONE=1.0E0)
      INTEGER IFLAG,LICN,N
      REAL A(LICN),W(N)
      INTEGER ICN(LICN),IDISP(2),IP(N),IQ(N),IW(N),LENR(N),LENRL(N)
      REAL CNTL(5), RINFO(5)
      INTEGER ICNTL(10)
      REAL AU,ROWMAX
      INTEGER I,IFIN,ILEND,IPIVJ,ISING,ISTART,J,JAY,JAYJAY,JFIN,JJ,
     +        PIVPOS
      LOGICAL STAB
      STAB = CNTL(1) .LE. ONE
      ISING = 0
      IFLAG = 0
      DO 10 I = 1,N
        W(I) = ZERO
   10 CONTINUE
      IW(1) = IDISP(1)
      IF (N.EQ.1) GO TO 25
      DO 20 I = 2,N
        IW(I) = IW(I-1) + LENR(I-1)
   20 CONTINUE
   25 CONTINUE
      DO 160 I = 1,N
        ISTART = IW(I)
        IFIN = ISTART + LENR(I) - 1
        ILEND = ISTART + LENRL(I) - 1
        IF (ISTART.GT.ILEND) GO TO 90
        DO 30 JJ = ISTART,IFIN
          J = ICN(JJ)
          W(J) = A(JJ)
   30   CONTINUE
        DO 70 JJ = ISTART,ILEND
          J = ICN(JJ)
          IPIVJ = IW(J) + LENRL(J)
          AU = -W(J)/A(IPIVJ)
          IF (ICNTL(6).NE.0) RINFO(2) = MAX(ABS(AU),RINFO(2))
          W(J) = AU
          IPIVJ = IPIVJ + 1
          JFIN = IW(J) + LENR(J) - 1
          IF (IPIVJ.GT.JFIN) GO TO 70
          IF (ICNTL(6).NE.0) GO TO 50
          DO 40 JAYJAY = IPIVJ,JFIN
            JAY = ICN(JAYJAY)
            W(JAY) = W(JAY) + AU*A(JAYJAY)
   40     CONTINUE
          GO TO 70
   50     CONTINUE
          DO 60 JAYJAY = IPIVJ,JFIN
            JAY = ICN(JAYJAY)
            W(JAY) = W(JAY) + AU*A(JAYJAY)
            RINFO(2) = MAX(ABS(W(JAY)),RINFO(2))
   60     CONTINUE
   70   CONTINUE
        DO 80 JJ = ISTART,IFIN
          J = ICN(JJ)
          A(JJ) = W(J)
          W(J) = ZERO
   80   CONTINUE
   90   CONTINUE
        PIVPOS = ILEND + 1
        IF (IQ(I).GT.0) GO TO 140
        IF (ISING.EQ.0) ISING = I
        IF (PIVPOS.GT.IFIN) GO TO 100
        IF (A(PIVPOS).NE.ZERO) GO TO 170
  100   CONTINUE
        IF (ISTART.GT.IFIN) GO TO 120
        DO 110 JJ = ISTART,IFIN
          IF (ICN(JJ).LT.ISING) GO TO 110
          IF (A(JJ).NE.ZERO) GO TO 170
  110   CONTINUE
  120   CONTINUE
        IF (PIVPOS.LE.IFIN) A(PIVPOS) = ONE
        IF (IP(I).GT.0 .AND. I.NE.N) GO TO 160
        DO 130 J = ISING,I
          IF ((LENR(J)-LENRL(J)).EQ.0) GO TO 130
          JJ = IW(J) + LENRL(J)
          A(JJ) = ZERO
  130   CONTINUE
        ISING = 0
        GO TO 160
  140   CONTINUE
        IF (PIVPOS.GT.IFIN) GO TO 170
        IF (A(PIVPOS).EQ.ZERO) GO TO 170
        IF (.NOT.STAB) GO TO 160
        ROWMAX = ZERO
        DO 150 JJ = PIVPOS,IFIN
          ROWMAX = MAX(ROWMAX,ABS(A(JJ)))
  150   CONTINUE
        IF (ABS(A(PIVPOS))/ROWMAX.GE.RINFO(1)) GO TO 160
        IFLAG = I
        RINFO(1) = ABS(A(PIVPOS))/ROWMAX
  160 CONTINUE
      GO TO 180
  170 IF (ICNTL(1).GT.0) WRITE (ICNTL(1),FMT=99999) I
      IFLAG = -I
  180 RETURN
99999 FORMAT (' Error return from MA33B/BD singularity detected in',
     +       ' row',I8)
      END
      SUBROUTINE MA33C(N,ICN,A,LICN,LENR,LENRL,LENOFF,IDISP,IP,IQ,X,W,
     +                  MTYPE,RINFO)
      INTEGER, INTENT( IN ) :: n, licn, mtype
      INTEGER, INTENT( IN ), DIMENSION( licn ) :: ICN
      INTEGER, INTENT( IN ), DIMENSION( n ) :: LENR, LENRL, LENOFF
      INTEGER, INTENT( IN ), DIMENSION( 2 ) :: IDISP
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: IP, IQ
      REAL, INTENT( IN ), DIMENSION( licn ) :: A
      REAL, INTENT( OUT ), DIMENSION( n ) :: W
      REAL, INTENT( INOUT ), DIMENSION( n ) :: X
      REAL, INTENT( INOUT ), DIMENSION( 5 ) :: RINFO
      REAL ZERO
      PARAMETER (ZERO=0.0E0)
      REAL WI,WII
      INTEGER I,IB,IBACK,IBLEND,IBLOCK,IEND,IFIRST,II,III,ILAST,J,J1,J2,
     +        J3,JJ,JPIV,JPIVP1,K,LJ1,LJ2,LT,LTEND,NUMBLK
      LOGICAL NEG,NOBLOC
      RINFO(3) = ZERO
      NOBLOC = LENOFF(1) .LT. 0
      IF (MTYPE.NE.1) GO TO 140
      NEG = .FALSE.
      IP(N) = -IP(N)
      DO 10 II = 1,N
        I = IP(II)
        I = ABS(I)
        W(II) = X(I)
   10 CONTINUE
      LT = 1
      IFIRST = 1
      IBLOCK = IDISP(1)
      DO 120 I = 1,N
        WI = W(I)
        IF (NOBLOC) GO TO 30
        IF (LENOFF(I).EQ.0) GO TO 30
        LTEND = LT + LENOFF(I) - 1
        DO 20 JJ = LT,LTEND
          J = ICN(JJ)
          WI = WI - A(JJ)*W(J)
   20   CONTINUE
        LT = LTEND + 1
   30   CONTINUE
        IF (IP(I).LT.0) NEG = .TRUE.
        IF (LENRL(I).EQ.0) GO TO 50
        IEND = IBLOCK + LENRL(I) - 1
        DO 40 JJ = IBLOCK,IEND
          J = ICN(JJ)
          WI = WI + A(JJ)*W(J)
   40   CONTINUE
   50   CONTINUE
        IBLOCK = IBLOCK + LENR(I)
        W(I) = WI
        IF (.NOT.NEG) GO TO 120
        J1 = IBLOCK
        IB = I
        IF (IQ(I).GT.0) GO TO 70
        DO 60 III = IFIRST,I
          IB = I - III + IFIRST
          IF (IQ(IB).GT.0) GO TO 70
          J1 = J1 - LENR(IB)
          RINFO(3) = MAX(RINFO(3),ABS(W(IB)))
          W(IB) = ZERO
   60   CONTINUE
        GO TO 110
   70   CONTINUE
        DO 100 III = IFIRST,IB
          II = IB - III + IFIRST
          J2 = J1 - 1
          J1 = J1 - LENR(II)
          JPIV = J1 + LENRL(II)
          JPIVP1 = JPIV + 1
          IF (J2.LT.JPIVP1) GO TO 90
          WII = W(II)
          DO 80 JJ = JPIVP1,J2
            J = ICN(JJ)
            WII = WII - A(JJ)*W(J)
   80     CONTINUE
          W(II) = WII
   90     CONTINUE
          W(II) = W(II)/A(JPIV)
  100   CONTINUE
  110   CONTINUE
        IFIRST = I + 1
        NEG = .FALSE.
  120 CONTINUE
      DO 130 II = 1,N
        I = IQ(II)
        I = ABS(I)
        X(I) = W(II)
  130 CONTINUE
      IP(N) = -IP(N)
      GO TO 320
  140 CONTINUE
      DO 150 II = 1,N
        I = IQ(II)
        I = ABS(I)
        W(II) = X(I)
  150 CONTINUE
      LJ1 = IDISP(1)
      IBLOCK = IDISP(2) + 1
      ILAST = N
      IBLEND = IBLOCK
      DO 290 NUMBLK = 1,N
        IF (ILAST.EQ.0) GO TO 300
        IBLOCK = IBLOCK - LENR(ILAST)
        DO 160 K = 1,N
          II = ILAST - K
          IF (II.EQ.0) GO TO 170
          IF (IP(II).LT.0) GO TO 170
          IBLOCK = IBLOCK - LENR(II)
  160   CONTINUE
  170   CONTINUE
        IFIRST = II + 1
        J1 = IBLOCK
        DO 210 I = IFIRST,ILAST
          IF (W(I).EQ.ZERO) GO TO 200
          IF (IQ(I).LT.0) GO TO 220
          J2 = J1 + LENRL(I)
          WI = W(I)/A(J2)
          IF (LENR(I)-LENRL(I).EQ.1) GO TO 190
          J2 = J2 + 1
          J3 = J1 + LENR(I) - 1
          DO 180 JJ = J2,J3
            J = ICN(JJ)
            W(J) = W(J) - A(JJ)*WI
  180     CONTINUE
  190     CONTINUE
          W(I) = WI
  200     CONTINUE
          J1 = J1 + LENR(I)
  210   CONTINUE
        GO TO 240
  220   CONTINUE
        DO 230 II = I,ILAST
          RINFO(3) = MAX(RINFO(3),ABS(W(II)))
          W(II) = ZERO
  230   CONTINUE
  240   CONTINUE
        J1 = IBLEND
        DO 280 IBACK = IFIRST,ILAST
          I = ILAST - IBACK + IFIRST
          J1 = J1 - LENR(I)
          IF (LENRL(I).EQ.0) GO TO 260
          J2 = J1 + LENRL(I) - 1
          DO 250 JJ = J1,J2
            J = ICN(JJ)
            W(J) = W(J) + A(JJ)*W(I)
  250     CONTINUE
  260     CONTINUE
          IF (NOBLOC) GO TO 280
          IF (LENOFF(I).EQ.0) GO TO 280
          LJ2 = LJ1 - 1
          LJ1 = LJ1 - LENOFF(I)
          DO 270 JJ = LJ1,LJ2
            J = ICN(JJ)
            W(J) = W(J) - A(JJ)*W(I)
  270     CONTINUE
  280   CONTINUE
        IBLEND = J1
        ILAST = IFIRST - 1
  290 CONTINUE
  300 DO 310 II = 1,N
        I = IP(II)
        I = ABS(I)
        X(I) = W(II)
  310 CONTINUE
  320 RETURN
      END
      SUBROUTINE MA33D(A,ICN,IPTR,N,IACTIV,ITOP,REALS,INFO)
      INTEGER IACTIV,ITOP,N
      LOGICAL REALS
      REAL A(ITOP)
      INTEGER ICN(ITOP),IPTR(N)
      INTEGER INFO(10)
      INTEGER J,JPOS,K,KL,KN
      IF (REALS) INFO(2) = INFO(2) + 1
      IF (.NOT.REALS) INFO(1) = INFO(1) + 1
      DO 10 J = 1,N
        K = IPTR(J)
        IF (K.LT.IACTIV) GO TO 10
        IPTR(J) = ICN(K)
        ICN(K) = -J
   10 CONTINUE
      KN = ITOP + 1
      KL = ITOP - IACTIV + 1
      DO 30 K = 1,KL
        JPOS = ITOP - K + 1
        IF (ICN(JPOS).EQ.0) GO TO 30
        KN = KN - 1
        IF (REALS) A(KN) = A(JPOS)
        IF (ICN(JPOS).GE.0) GO TO 20
          J = -ICN(JPOS)
          ICN(JPOS) = IPTR(J)
          IPTR(J) = KN
   20   CONTINUE
        ICN(KN) = ICN(JPOS)
   30 CONTINUE
      IACTIV = KN
      RETURN
      END
