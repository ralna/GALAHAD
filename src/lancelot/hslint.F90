! THIS VERSION: GALAHAD 4.1 - 2022-12-18 AT 10:15 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-  L A N C E L O T  -B-  HSL_routines  M O D U L E  *-*-*-*-*-*

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  February 5th 1995

   MODULE LANCELOT_HSL_routines
            
     USE GALAHAD_PRECISION

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: MA27_initialize, MA27_analyse, MA27_factorize, MA27_solve,      &
               MA61_initialize, MA61_compress

!  Define generic interfaces to HSL routines

     INTERFACE MA27_initialize
       SUBROUTINE MA27I( ICNTL, CNTL )
       USE GALAHAD_PRECISION
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 30 ) :: ICNTL
       REAL ( KIND = sp_), INTENT( OUT ), DIMENSION( 5 ) :: CNTL
       END SUBROUTINE MA27I

       SUBROUTINE MA27ID( ICNTL, CNTL )
       USE GALAHAD_PRECISION
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 30 ) :: ICNTL
       REAL ( KIND = dp_), INTENT( OUT ), DIMENSION( 5 ) :: CNTL
       END SUBROUTINE MA27ID
     END INTERFACE

     INTERFACE MA27_analyse
       SUBROUTINE MA27A( n, nz, IRN, ICN, IW, liw, IKEEP, IW1, nsteps,         &
                         iflag, ICNTL, CNTL, INFO, ops )
       USE GALAHAD_PRECISION
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nz, liw
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: nsteps
       INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iflag
       REAL( KIND = sp_), INTENT( OUT ) :: ops
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( liw ) :: IW
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n, 3 ) :: IKEEP
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n, 2 ) :: IW1
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 20 ) :: INFO
       REAL ( KIND = sp_), INTENT( IN ), DIMENSION( 5 ) :: CNTL
       END SUBROUTINE MA27A

       SUBROUTINE MA27AD( n, nz, IRN, ICN, IW, liw, IKEEP, IW1, nsteps,        &
                          iflag, ICNTL, CNTL, INFO, ops )
       USE GALAHAD_PRECISION
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nz, liw
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: nsteps
       INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iflag
       REAL( KIND = dp_), INTENT( OUT ) :: ops
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( liw ) :: IW
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n, 3 ) :: IKEEP
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n, 2 ) :: IW1
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 20 ) :: INFO
       REAL ( KIND = dp_), INTENT( IN ), DIMENSION( 5 ) :: CNTL
       END SUBROUTINE MA27AD
     END INTERFACE

     INTERFACE MA27_factorize
       SUBROUTINE MA27B( n, nz, IRN, ICN, A, la, IW, liw, IKEEP, nsteps,       &
                         maxfrt, IW1, ICNTL, CNTL, INFO )
       USE GALAHAD_PRECISION
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nz, la, liw, nsteps
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: maxfrt
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( liw ) :: IW
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n, 3 ) :: IKEEP
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: IW1
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 20 ) :: INFO
       REAL ( KIND = sp_), INTENT( IN ), DIMENSION( 5 ) :: CNTL
       REAL ( KIND = sp_), INTENT( INOUT ), DIMENSION( la ) :: A
       END SUBROUTINE MA27B

       SUBROUTINE MA27BD( n, nz, IRN, ICN, A, la, IW, liw, IKEEP, nsteps,      &
                          maxfrt, IW1, ICNTL, CNTL, INFO )
       USE GALAHAD_PRECISION
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nz, la, liw, nsteps
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: maxfrt
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( liw ) :: IW
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n, 3 ) :: IKEEP
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: IW1
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 20 ) :: INFO
       REAL ( KIND = dp_), INTENT( IN ), DIMENSION( 5 ) :: CNTL
       REAL ( KIND = dp_), INTENT( INOUT ), DIMENSION( la ) :: A
       END SUBROUTINE MA27BD
     END INTERFACE

     INTERFACE MA27_solve
       SUBROUTINE MA27C( n, A, la, IW, liw, W, maxfrt, RHS, IW1, nsteps,       &
                         ICNTL, INFO )
       USE GALAHAD_PRECISION
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, la, liw, maxfrt, nsteps
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( liw ) :: IW
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( nsteps ) :: IW1
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 20 ) :: INFO
       REAL( KIND = sp_), INTENT( IN ), DIMENSION( la ) :: A
       REAL( KIND = sp_), INTENT( OUT ), DIMENSION( maxfrt ) :: W
       REAL( KIND = sp_), INTENT( INOUT ), DIMENSION( n ) :: RHS
       END SUBROUTINE MA27C

       SUBROUTINE MA27CD( n, A, la, IW, liw, W, maxfrt, RHS, IW1, nsteps,      &
                          ICNTL, INFO )
       USE GALAHAD_PRECISION
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, la, liw, maxfrt, nsteps
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( liw ) :: IW
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( nsteps ) :: IW1
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 20 ) :: INFO
       REAL( KIND = dp_), INTENT( IN ), DIMENSION( la ) :: A
       REAL( KIND = dp_), INTENT( OUT ), DIMENSION( maxfrt ) :: W
       REAL( KIND = dp_), INTENT( INOUT ), DIMENSION( n ) :: RHS
       END SUBROUTINE MA27CD
     END INTERFACE

     INTERFACE MA61_initialize
       SUBROUTINE MA61I( ICNTL, CNTL, KEEP )
       USE GALAHAD_PRECISION
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: ICNTL( 5 ), KEEP( 12 )
       REAL ( KIND = sp_), INTENT( OUT ) :: CNTL( 3 )
       END SUBROUTINE MA61I

       SUBROUTINE MA61ID( ICNTL, CNTL, KEEP )
       USE GALAHAD_PRECISION
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: ICNTL( 5 ), KEEP( 12 )
       REAL ( KIND = dp_), INTENT( OUT ) :: CNTL( 3 )
       END SUBROUTINE MA61ID
     END INTERFACE

     INTERFACE MA61_compress
       SUBROUTINE MA61D( A, IRN, ia, n, IK, IP, row, ncp, nucl, nual )
       USE GALAHAD_PRECISION
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: ia, n
       INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: ncp, nucl, nual
       LOGICAL, INTENT( IN ) :: row
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( ia ) :: IRN
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: IK
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: IP
       REAL ( KIND = sp_), INTENT( INOUT ), DIMENSION( ia ) :: A
       END SUBROUTINE MA61D

       SUBROUTINE MA61DD( A, IRN, ia, n, IK, IP, row, ncp, nucl, nual )
       USE GALAHAD_PRECISION
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: ia, n
       INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: ncp, nucl, nual
       LOGICAL, INTENT( IN ) :: row
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( ia ) :: IRN
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: IK
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: IP
       REAL ( KIND = dp_), INTENT( INOUT ), DIMENSION( ia ) :: A
       END SUBROUTINE MA61DD
     END INTERFACE

!  End of module LANCELOT_HSL_routines

   END MODULE LANCELOT_HSL_routines
