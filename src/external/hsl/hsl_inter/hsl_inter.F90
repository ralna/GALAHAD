! THIS VERSION: GALAHAD 5.1 - 2024-11-23 AT 10:45 GMT

#include "galahad_modules.h"
#include "hsl_subset.h"

!-*-*-*-*-*-  G A L A H A D _ H S L _ i n t e r    M O D U L E  -*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 5.1. November 18th 2024

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_HSL_inter_precision

      IMPLICIT NONE

      PUBLIC

!--------------------------------------------------------------------
!   I n t e r f a c e  B l o c k s  t o  H S L  S u b r o u t i n e s
!--------------------------------------------------------------------

!  LA04 subroutines

      INTERFACE LA04A

        SUBROUTINE LA04AR( A, la, IRN, IP, m, n, B, C, BND, kb, lb, job, CNTL, &
                           IX, JX, X, Z, G, RINFO, WS, lws, IWS, liws )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: la, m, n, kb, lb, lws, liws
        INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: job
        INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n + 1 ) :: IP
        INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( la ) :: IRN
        INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( m ) :: IX
        INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( kb ) :: JX
        INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( liws ) :: IWS
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( la ) :: A
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( m ) :: B
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( 2, kb ) :: BND
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: C
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( 15 ) :: CNTL
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( 40 ) :: RINFO
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n + m ) :: X
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: Z
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: G
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( lws ) :: WS
        END SUBROUTINE LA04AR

      END INTERFACE LA04A

!  MA27 subroutines

     INTERFACE MA27A

       SUBROUTINE MA27AR( n, nz, IRN, ICN, IW, liw, IKEEP, IW2, nsteps,        &
                          iflag, ICNTL, CNTL, INFO, ops )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nz, liw
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: nsteps
       INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iflag
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( liw ) :: IW
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n, 3 ) :: IKEEP
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n, 2 ) :: IW2
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 20 ) :: INFO
       REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( 5 ) :: CNTL
       REAL ( KIND = rp_ ), INTENT( OUT ) :: ops
       END SUBROUTINE MA27AR

     END INTERFACE MA27A

     INTERFACE MA27B

       SUBROUTINE MA27BR( n, nz, IRN, ICN, A, la, IW, liw, IKEEP, nsteps,      &
                          maxfrt, IW1, ICNTL, CNTL, INFO )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nz, la, liw, nsteps
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: maxfrt
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( liw ) :: IW
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n, 3 ) :: IKEEP
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: IW1
       REAL( KIND = rp_ ), INTENT( INOUT ), DIMENSION( la ) :: A
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 20 ) :: INFO
       REAL( KIND = rp_ ), INTENT( IN ), DIMENSION( 5 ) :: CNTL
       END SUBROUTINE MA27BR

     END INTERFACE MA27B

     INTERFACE MA27C

       SUBROUTINE MA27CR( n, A, la, IW, liw, W, maxfrt, RHS, IW1, nsteps,      &
                          ICNTL, INFO )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, la, liw, maxfrt, nsteps
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( liw ) :: IW
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( nsteps ) :: IW1
       REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( la ) :: A
       REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( maxfrt ) :: W
       REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: RHS
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 20 ) :: INFO
       END SUBROUTINE MA27CR

     END INTERFACE MA27C

     INTERFACE MA27I

       SUBROUTINE MA27IR( ICNTL, CNTL )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: ICNTL( 30 )
       REAL ( KIND = rp_ ), INTENT( OUT ) :: CNTL( 5 )
       END SUBROUTINE MA27IR

     END INTERFACE MA27I

     INTERFACE MA27Q

       SUBROUTINE MA27QR( n, A, la, IW, liw, W, maxfnt, RHS, IW2, nblk,        &
                          latop, ICNTL )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, la, liw, maxfnt, nblk, latop
       REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( la ) :: A
       REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: RHS
       REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( maxfnt ) :: W
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( liw ) :: IW
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( nblk ) :: IW2
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       END SUBROUTINE MA27QR

     END INTERFACE MA27Q

!  MA33 subroutines

     INTERFACE MA33A

       SUBROUTINE MA33AR( n, ICN, A, licn, LENR, LENRL, IDISP, IP, IQ, IRN,    &
                          lirn, LENC, IFIRST, LASTR, NEXTR, LASTC, NEXTC,      &
                          IPTR, IPC, u, iflag, ICNTL, CNTL, INFO, RINFO )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, licn, lirn
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: iflag
       REAL ( KIND = rp_ ), INTENT( INOUT ) :: u
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( licn ) ::  ICN
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) ::  LENR
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) ::  LENRL
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( 2 ) ::  IDISP
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) ::  IP, IQ
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( lirn ) :: IRN
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) ::  IPC, IPTR
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) ::  LENC, IFIRST
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) ::  LASTR, NEXTR
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) ::  LASTC, NEXTC
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 10 ) :: INFO
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 10 ) :: ICNTL
       REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( licn ) :: A
       REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( 5 ) :: RINFO
       REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( 5 ) :: CNTL
       END SUBROUTINE MA33AR

     END INTERFACE MA33A

     INTERFACE MA33C

       SUBROUTINE MA33CR( n, ICN, A, licn, LENR, LENRL, LENOFF, IDISP,         &
                          IP, IQ, X, W, mtype, RINFO )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, licn, mtype
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( licn ) :: ICN
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n ) :: LENR, LENRL
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n ) :: LENOFF
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 2 ) :: IDISP
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: IP, IQ
       REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( licn ) :: A
       REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( n ) :: W
       REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X
       REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( 5 ) :: RINFO
       END SUBROUTINE MA33CR

     END INTERFACE MA33C

     INTERFACE MA33I

       SUBROUTINE MA33IR( ICNTL, CNTL )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 10 ) :: ICNTL
       REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( 5 ) :: CNTL
       END SUBROUTINE MA33IR

     END INTERFACE MA33I

!  MA61 subroutines

     INTERFACE MA61I

       SUBROUTINE MA61IR( ICNTL, CNTL, KEEP )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: ICNTL( 5 ), KEEP( 12 )
       REAL ( KIND = rp_), INTENT( OUT ) :: CNTL( 3 )
       END SUBROUTINE MA61IR

     END INTERFACE MA61I

     INTERFACE MA61D

       SUBROUTINE MA61DR( A, IRN, ia, n, IK, IP, row, ncp, nucl, nual )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: ia, n
       INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: ncp, nucl, nual
       LOGICAL, INTENT( IN ) :: row
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( ia ) :: IRN
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: IK
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: IP
       REAL ( KIND = rp_), INTENT( INOUT ), DIMENSION( ia ) :: A
       END SUBROUTINE MA61DR

     END INTERFACE MA61D

!  MC20 subroutines

     INTERFACE MC20A

       SUBROUTINE MC20AR( nc, maxa, A, INUM, JPTR, JNUM, jdisp )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: nc, maxa, jdisp
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( maxa ) :: INUM, JNUM
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( nc ) :: JPTR
       REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( maxa ) :: A
       END SUBROUTINE MC20AR

     END INTERFACE MC20A

!  MC61 subroutines

     INTERFACE MC61A

       SUBROUTINE MC61AR( job, n, lirn, IRN, ICPTR, PERM, liw, IW, W,          &
                          ICNTL, CNTL, INFO, RINFO )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: job, n, liw, lirn
       INTEGER ( KIND = ip_ ), DIMENSION( lirn ), INTENT( INOUT ) :: IRN
       INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ), INTENT( INOUT ) :: ICPTR
       INTEGER ( KIND = ip_ ), DIMENSION( n ), INTENT( INOUT ) :: PERM
       INTEGER ( KIND = ip_ ), DIMENSION( liw ), INTENT( OUT ) :: IW
       REAL ( KIND = rp_ ), DIMENSION( n ), INTENT( OUT ) :: W
       INTEGER ( KIND = ip_ ), DIMENSION( 10 ), INTENT( IN ) :: ICNTL
       REAL ( KIND = rp_ ), DIMENSION( 5 ), INTENT( IN ) :: CNTL
       INTEGER ( KIND = ip_ ), DIMENSION( 10 ), INTENT( OUT ) :: INFO
       REAL ( KIND = rp_ ), DIMENSION( 15 ), INTENT( OUT ) :: RINFO
       END SUBROUTINE MC61AR

     END INTERFACE MC61A

!  MC77 subroutines

    INTERFACE MC77A

      SUBROUTINE MC77AR( job, m, n, nnz, JCST, IRN, A, IW, liw, DW, ldw,       &
                         ICNTL, CNTL, INFO, RINFO )
      USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: job, m, n, nnz, liw, ldw
      INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ), INTENT( IN ) :: JCST
      INTEGER ( KIND = ip_ ), DIMENSION( nnz ), INTENT( IN ) :: IRN
      INTEGER ( KIND = ip_ ), DIMENSION( liw ), INTENT( OUT ) :: IW
      REAL ( KIND = rp_ ), DIMENSION( nnz ), INTENT( IN ) :: A
      REAL ( KIND = rp_ ), DIMENSION( ldw ), INTENT( OUT ) :: DW
      INTEGER ( KIND = ip_ ), DIMENSION( 10 ), INTENT( IN ) :: ICNTL
      REAL ( KIND = rp_ ), DIMENSION( 10 ), INTENT( IN ) :: CNTL
      INTEGER ( KIND = ip_ ), DIMENSION( 10 ), INTENT( OUT ) :: INFO
      REAL ( KIND = rp_ ), DIMENSION( 10 ), INTENT( OUT ) :: RINFO
      END SUBROUTINE MC77AR

    END INTERFACE MC77A

    END MODULE GALAHAD_HSL_inter_precision
