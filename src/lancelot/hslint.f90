! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
! Updated 09/01/2004: correct intent for INFO and IFLAG for MA27 given

!-*-*-*-*-*-  L A N C E L O T  -B-  HSL_routines  M O D U L E  *-*-*-*-*-*

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  February 5th 1995

   MODULE LANCELOT_HSL_routines

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: MA27_initialize, MA27_analyse, MA27_factorize, MA27_solve,      &
               MA61_initialize, MA61_compress

!  Define generic interfaces to HSL routines

     INTERFACE MA27_initialize

       SUBROUTINE MA27I ( ICNTL, CNTL )
       INTEGER, INTENT( OUT ), DIMENSION( 30 ) :: ICNTL
       REAL ( KIND = KIND( 1.0E0 ) ), INTENT( OUT ), DIMENSION( 5 ) :: CNTL
       END SUBROUTINE MA27I

       SUBROUTINE MA27ID( ICNTL, CNTL )
       INTEGER, INTENT( OUT ), DIMENSION( 30 ) :: ICNTL
       REAL ( KIND = KIND( 1.0D0 ) ), INTENT( OUT ), DIMENSION( 5 ) :: CNTL
       END SUBROUTINE MA27ID

     END INTERFACE

     INTERFACE MA27_analyse

       SUBROUTINE MA27A ( n, nz, IRN, ICN, IW, liw, IKEEP, IW1, nsteps,        &
                          iflag, ICNTL, CNTL, INFO, ops )
       INTEGER, INTENT( IN ) :: n, nz, liw
       INTEGER, INTENT( OUT ) :: nsteps
       INTEGER, INTENT( INOUT ) :: iflag
       REAL( KIND = KIND( 1.0E0 ) ), INTENT( OUT ) :: ops
       INTEGER, INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
       INTEGER, INTENT( OUT ), DIMENSION( liw ) :: IW
       INTEGER, INTENT( INOUT ), DIMENSION( 3 * n ) :: IKEEP
       INTEGER, INTENT( OUT ), DIMENSION( 2 * n ) :: IW1
       INTEGER, INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER, INTENT( OUT ), DIMENSION( 20 ) :: INFO
       REAL ( KIND = KIND( 1.0E0 ) ), INTENT( IN ), DIMENSION( 5 ) :: CNTL
       END SUBROUTINE MA27A

       SUBROUTINE MA27AD( n, nz, IRN, ICN, IW, liw, IKEEP, IW1, nsteps,        &
                          iflag, ICNTL, CNTL, INFO, ops )
       INTEGER, INTENT( IN ) :: n, nz, liw
       INTEGER, INTENT( OUT ) :: nsteps
       INTEGER, INTENT( INOUT ) :: iflag
       REAL( KIND = KIND( 1.0D0 ) ), INTENT( OUT ) :: ops
       INTEGER, INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
       INTEGER, INTENT( OUT ), DIMENSION( liw ) :: IW
       INTEGER, INTENT( INOUT ), DIMENSION( 3 * n ) :: IKEEP
       INTEGER, INTENT( OUT ), DIMENSION( 2 * n ) :: IW1
       INTEGER, INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER, INTENT( OUT ), DIMENSION( 20 ) :: INFO
       REAL ( KIND = KIND( 1.0D0 ) ), INTENT( IN ), DIMENSION( 5 ) :: CNTL
       END SUBROUTINE MA27AD

     END INTERFACE

     INTERFACE MA27_factorize

       SUBROUTINE MA27B ( n, nz, IRN, ICN, A, la, IW, liw, IKEEP, nsteps,      &
                          maxfrt, IW1, ICNTL, CNTL, INFO )
       INTEGER, INTENT( IN ) :: n, nz, la, liw, nsteps
       INTEGER, INTENT( OUT ) :: maxfrt
       INTEGER, INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
       INTEGER, INTENT( OUT ), DIMENSION( liw ) :: IW
       INTEGER, INTENT( IN ), DIMENSION( 3 * n ) :: IKEEP
       INTEGER, INTENT( OUT ), DIMENSION( n ) :: IW1
       INTEGER, INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER, INTENT( OUT ), DIMENSION( 20 ) :: INFO
       REAL ( KIND = KIND( 1.0E0 ) ), INTENT( IN ), DIMENSION( 5 ) :: CNTL
       REAL ( KIND = KIND( 1.0E0 ) ), INTENT( INOUT ), DIMENSION( la ) :: A
       END SUBROUTINE MA27B

       SUBROUTINE MA27BD( n, nz, IRN, ICN, A, la, IW, liw, IKEEP, nsteps,      &
                          maxfrt, IW1, ICNTL, CNTL, INFO )
       INTEGER, INTENT( IN ) :: n, nz, la, liw, nsteps
       INTEGER, INTENT( OUT ) :: maxfrt
       INTEGER, INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
       INTEGER, INTENT( OUT ), DIMENSION( liw ) :: IW
       INTEGER, INTENT( IN ), DIMENSION( 3 * n ) :: IKEEP
       INTEGER, INTENT( OUT ), DIMENSION( n ) :: IW1
       INTEGER, INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER, INTENT( OUT ), DIMENSION( 20 ) :: INFO
       REAL ( KIND = KIND( 1.0D0 ) ), INTENT( IN ), DIMENSION( 5 ) :: CNTL
       REAL ( KIND = KIND( 1.0D0 ) ), INTENT( INOUT ), DIMENSION( la ) :: A
       END SUBROUTINE MA27BD

     END INTERFACE

     INTERFACE MA27_solve

       SUBROUTINE MA27C ( n, A, la, IW, liw, W, maxfrt, RHS, IW1, nsteps,      &
                          ICNTL, INFO )
       INTEGER, INTENT( IN ) :: n, la, liw, maxfrt, nsteps
       INTEGER, INTENT( IN ), DIMENSION( liw ) :: IW
       INTEGER, INTENT( OUT ), DIMENSION( nsteps ) :: IW1
       INTEGER, INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER, INTENT( OUT ), DIMENSION( 20 ) :: INFO
       REAL( KIND = KIND( 1.0E0 ) ), INTENT( IN ), DIMENSION( la ) :: A
       REAL( KIND = KIND( 1.0E0 ) ), INTENT( OUT ), DIMENSION( maxfrt ) :: W
       REAL( KIND = KIND( 1.0E0 ) ), INTENT( INOUT ), DIMENSION( n ) :: RHS
       END SUBROUTINE MA27C

       SUBROUTINE MA27CD( n, A, la, IW, liw, W, maxfrt, RHS, IW1, nsteps,      &
                          ICNTL, INFO )
       INTEGER, INTENT( IN ) :: n, la, liw, maxfrt, nsteps
       INTEGER, INTENT( IN ), DIMENSION( liw ) :: IW
       INTEGER, INTENT( OUT ), DIMENSION( nsteps ) :: IW1
       INTEGER, INTENT( IN ), DIMENSION( 30 ) :: ICNTL
       INTEGER, INTENT( OUT ), DIMENSION( 20 ) :: INFO
       REAL( KIND = KIND( 1.0D0 ) ), INTENT( IN ), DIMENSION( la ) :: A
       REAL( KIND = KIND( 1.0D0 ) ), INTENT( OUT ), DIMENSION( maxfrt ) :: W
       REAL( KIND = KIND( 1.0D0 ) ), INTENT( INOUT ), DIMENSION( n ) :: RHS
       END SUBROUTINE MA27CD

     END INTERFACE

     INTERFACE MA61_initialize

       SUBROUTINE MA61I ( ICNTL, CNTL, KEEP )
       INTEGER, INTENT( OUT ) :: ICNTL( 5 ), KEEP( 12 )
       REAL ( KIND = KIND( 1.0E0 ) ), INTENT( OUT ) :: CNTL( 3 )
       END SUBROUTINE MA61I

       SUBROUTINE MA61ID( ICNTL, CNTL, KEEP )
       INTEGER, INTENT( OUT ) :: ICNTL( 5 ), KEEP( 12 )
       REAL ( KIND = KIND( 1.0D0 ) ), INTENT( OUT ) :: CNTL( 3 )
       END SUBROUTINE MA61ID

     END INTERFACE 

     INTERFACE MA61_compress

       SUBROUTINE MA61D ( A, IRN, ia, n, IK, IP, row, ncp, nucl, nual )
       INTEGER, INTENT( IN ) :: ia, n
       INTEGER, INTENT( INOUT ) :: ncp, nucl, nual
       LOGICAL, INTENT( IN ) :: row
       INTEGER, INTENT( INOUT ), DIMENSION( ia ) :: IRN
       INTEGER, INTENT( INOUT ), DIMENSION( n ) :: IK
       INTEGER, INTENT( INOUT ), DIMENSION( n ) :: IP
       REAL ( KIND = KIND( 1.0E0 ) ), INTENT( INOUT ), DIMENSION( ia ) :: A
       END SUBROUTINE MA61D
     
       SUBROUTINE MA61DD( A, IRN, ia, n, IK, IP, row, ncp, nucl, nual )
       INTEGER, INTENT( IN ) :: ia, n
       INTEGER, INTENT( INOUT ) :: ncp, nucl, nual
       LOGICAL, INTENT( IN ) :: row
       INTEGER, INTENT( INOUT ), DIMENSION( ia ) :: IRN
       INTEGER, INTENT( INOUT ), DIMENSION( n ) :: IK
       INTEGER, INTENT( INOUT ), DIMENSION( n ) :: IP
       REAL ( KIND = KIND( 1.0D0 ) ), INTENT( INOUT ), DIMENSION( ia ) :: A
       END SUBROUTINE MA61DD

     END INTERFACE 

!  End of module LANCELOT_HSL_routines

   END MODULE LANCELOT_HSL_routines


