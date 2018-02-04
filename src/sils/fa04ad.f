* *******************************************************************
* COPYRIGHT (c) 1993 Hyprotech UK
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
*######DATE 25 Jan 1993
C       Toolpack tool decs employed.
C       SAVE statements added.
C       Reference to FA04FD removed.
C       DATA statements changed to PARAMETER.
C  EAT 21/6/93 EXTERNAL statement put in for block data on VAXs.
C
C
      DOUBLE PRECISION FUNCTION FA04AD(I)
      INTEGER A,B15,B16,P
      PARAMETER (A=16807,B15=32768,B16=65536,P=2147483647)
      INTEGER I
      DOUBLE PRECISION X
      INTEGER FHI,K,LEFTLO,XALO,XHI
      INTRINSIC FLOAT
      COMMON /FA04ED/IX
      INTEGER IX
      EXTERNAL FA04FD
      SAVE /FA04ED/
      XHI = IX/B16
      XALO = (IX-XHI*B16)*A
      LEFTLO = XALO/B16
      FHI = XHI*A + LEFTLO
      K = FHI/B15
      IX = (((XALO-LEFTLO*B16)-P)+ (FHI-K*B15)*B16) + K
      IF (IX.LT.0) IX = IX + P
      XHI = IX/B16
      X = (FLOAT(XHI)*65536.0D0) + FLOAT(IX-XHI*B16)
      IF (I.GE.0) FA04AD = X*4.6566128752457969241D-10
      IF (I.LT.0) FA04AD = X*9.3132257504915938482D-10 - 1.0D0
      RETURN
      END
      SUBROUTINE FA04BD(MAX,NRAND)
      INTEGER A,B15,B16,P
      PARAMETER (A=16807,B15=32768,B16=65536,P=2147483647)
      INTEGER B30,Q
      PARAMETER (B30=1073741824,Q=1073741823)
      INTEGER MAX,NRAND
      INTEGER BE1,BE2,C,D,F,FHI,G,K,LEFTLO,MHI,MLO,MU,NU,XALO,XHI,XLO
      COMMON /FA04ED/IX
      INTEGER IX
      SAVE /FA04ED/
      XHI = IX/B16
      XALO = (IX-XHI*B16)*A
      LEFTLO = XALO/B16
      FHI = XHI*A + LEFTLO
      K = FHI/B15
      IX = (((XALO-LEFTLO*B16)-P)+ (FHI-K*B15)*B16) + K
      IF (IX.LT.0) IX = IX + P
      XHI = IX/B15
      XLO = IX - B15*XHI
      MHI = MAX/B15
      MLO = MAX - B15*MHI
      F = (XHI*MLO-P) + XLO*MHI
      IF (F.GT.0) GO TO 1
      F = F + P
      BE1 = F/B15
      BE2 = F - BE1*B15
      GO TO 2
    1 F = F - 1
      BE1 = F/B15
      BE2 = F - BE1*B15
      BE1 = BE1 + B16
    2 G = B15*BE2 + XLO*MLO
      D = G/B30
      C = XHI/2
      F = ((2* (C*MHI-Q)-1)+MHI* (XHI-2*C)) + D + BE1
      IF (F.GT.0) GO TO 3
      F = F + P
      NU = F/2
      MU = F - NU*2
      GO TO 4
    3 F = F - 1
      NU = F/2
      MU = F - 2*NU
      NU = NU + B30
    4 F = (B30*MU-P) + NU + (G-B30*D)
      NRAND = NU + 1
      IF (F.GE.0) NRAND = NRAND + 1
      RETURN
      END
      SUBROUTINE FA04CD(IGEN)
      INTEGER IGEN
      COMMON /FA04ED/IX
      INTEGER IX
      SAVE /FA04ED/
      IGEN = IX
      RETURN
      END
      SUBROUTINE FA04DD(IGEN)
      INTEGER IGEN
      COMMON /FA04ED/IX
      INTEGER IX
      SAVE /FA04ED/
      IX = IGEN
      RETURN
      END
      BLOCK DATA FA04FD
      COMMON /FA04ED/IX
      INTEGER IX
      SAVE /FA04ED/
      DATA IX/1/
      END
