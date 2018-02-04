* *******************************************************************
* COPYRIGHT (c) 1975 Hyprotech UK
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
C       Array lengths given explicitly eg A(MAXA)
C
C  Modified for GALAHAD: INTENT added for all dummy arguments
C
      SUBROUTINE MC20AD(NC,MAXA,A,INUM,JPTR,JNUM,JDISP)
      INTEGER, INTENT( IN ) :: nc, maxa, jdisp
      INTEGER, INTENT( INOUT ), DIMENSION( maxa ) :: INUM, JNUM
      INTEGER, INTENT( OUT ), DIMENSION( nc ) :: JPTR
      DOUBLE PRECISION, INTENT( INOUT ), DIMENSION( maxa ) :: A
      DOUBLE PRECISION ACE,ACEP
      INTEGER I,ICE,ICEP,J,JA,JB,JCE,JCEP,K,KR,LOC,NULL
C     write(6,*) ' nc ', nc, ' maxa ', maxa
C     write(6,*) ' jnum ', JNUM( 1 : maxa )
      NULL = -JDISP
C**      CLEAR JPTR
      DO 10 J = 1,NC
        JPTR(J) = 0
   10 CONTINUE
C**      COUNT THE NUMBER OF ELEMENTS IN EACH COLUMN.
      DO 20 K = 1,MAXA
        J = JNUM(K) + JDISP
        JPTR(J) = JPTR(J) + 1
   20 CONTINUE
C**      SET THE JPTR ARRAY
      K = 1
      DO 30 J = 1,NC
        KR = K + JPTR(J)
        JPTR(J) = K
        K = KR
   30 CONTINUE
C**      REORDER THE ELEMENTS INTO COLUMN ORDER.  THE ALGORITHM IS AN
      DO 50 I = 1,MAXA
        JCE = JNUM(I) + JDISP
        IF (JCE.EQ.0) GO TO 50
        ACE = A(I)
        ICE = INUM(I)
        JNUM(I) = NULL
        DO 40 J = 1,MAXA
          LOC = JPTR(JCE)
          JPTR(JCE) = JPTR(JCE) + 1
          ACEP = A(LOC)
          ICEP = INUM(LOC)
          JCEP = JNUM(LOC)
          A(LOC) = ACE
          INUM(LOC) = ICE
          JNUM(LOC) = NULL
          IF (JCEP.EQ.NULL) GO TO 50
          ACE = ACEP
          ICE = ICEP
          JCE = JCEP + JDISP
   40   CONTINUE
   50 CONTINUE
C**      RESET JPTR VECTOR.
      JA = 1
      DO 60 J = 1,NC
        JB = JPTR(J)
        JPTR(J) = JA
        JA = JB
   60 CONTINUE
      RETURN
      END
      SUBROUTINE MC20BD(NC,MAXA,A,INUM,JPTR)
      INTEGER MAXA,NC
      DOUBLE PRECISION A(MAXA)
      INTEGER INUM(MAXA),JPTR(NC)
      DOUBLE PRECISION ACE
      INTEGER ICE,IK,J,JJ,K,KDUMMY,KLO,KMAX,KOR
      INTRINSIC IABS
      KMAX = MAXA
      DO 50 JJ = 1,NC
        J = NC + 1 - JJ
        KLO = JPTR(J) + 1
        IF (KLO.GT.KMAX) GO TO 40
        KOR = KMAX
        DO 30 KDUMMY = KLO,KMAX
          ACE = A(KOR-1)
          ICE = INUM(KOR-1)
          DO 10 K = KOR,KMAX
            IK = INUM(K)
            IF (IABS(ICE).LE.IABS(IK)) GO TO 20
            INUM(K-1) = IK
            A(K-1) = A(K)
   10     CONTINUE
          K = KMAX + 1
   20     INUM(K-1) = ICE
          A(K-1) = ACE
          KOR = KOR - 1
   30   CONTINUE
   40   KMAX = KLO - 2
   50 CONTINUE
      RETURN
      END
