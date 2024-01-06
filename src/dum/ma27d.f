! THIS VERSION: GALAHAD 4.3 - 2024-01-06 AT 07:50 GMT.

!-*-*-*-*-*-*-  G A L A H A D  -  MA27  S U B R O U T I N E S *-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  February 13th 1995

      SUBROUTINE MA27AD( n, nz, IRN, ICN, IW, liw, IKEEP, IW1, 
     *                   nsteps, iflag, ICNTL, CNTL, INFO, ops )

      USE GALAHAD_KINDS_double

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nz, liw
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iflag
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: nsteps
      REAL ( KIND = rp_ ), INTENT( OUT ) :: ops
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( liw ) :: IW
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n,3 ) :: IKEEP
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n, 2 ) :: IW1
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 20 ) :: INFO
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( 5 ) :: CNTL

!  Dummy subroutine available with GALAHAD

      IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) > 0 ) 
     *  WRITE ( ICNTL( 2 ), 2000 )
      INFO( 1 ) = - 26
      RETURN

!  Non-executable statements

 2000 FORMAT( /, 
     *     ' We regret that the solution options that you have ', /, 
     *     ' chosen are not all freely available with GALAHAD.', //,
     *     ' If you have HSL (formerly the Harwell Subroutine',
     *     ' Library), this ', /,
     *     ' option may be enabled by replacing the dummy ', /,      
     *     ' subroutine MA27AD with its HSL namesake ', /,        
     *     ' and dependencies. See ', /,
     *     '   $GALAHAD/src/makedefs/packages for details.', //,
     *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MA27AD

      END SUBROUTINE MA27AD


      SUBROUTINE MA27BD( n, nz, IRN, ICN, A, la, IW, liw, IKEEP, 
     *                   nsteps, maxfrt, IW1, ICNTL, CNTL, INFO )

      USE GALAHAD_KINDS_double

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nz, la, liw, nsteps
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: maxfrt
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( liw ) :: IW
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n, 3 ) :: IKEEP
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: IW1
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 20 ) :: INFO
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( 5 ) :: CNTL
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( la ) :: A

!  Dummy subroutine available with GALAHAD

      IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) > 0 ) 
     *  WRITE ( ICNTL( 2 ), 2000 )
      INFO( 1 ) = - 26
      RETURN

!  Non-executable statements

 2000 FORMAT( /, 
     *     ' We regret that the solution options that you have ', /, 
     *     ' chosen are not all freely available with GALAHAD.', //,
     *     ' If you have HSL (formerly the Harwell Subroutine',
     *     ' Library), this ', /,
     *     ' option may be enabled by replacing the dummy ', /,      
     *     ' subroutine MA27BD with its HSL namesake ', /,        
     *     ' and dependencies. See ', /,
     *     '   $GALAHAD/src/makedefs/packages for details.', //,
     *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MA27BD

      END SUBROUTINE MA27BD


      SUBROUTINE MA27CD( n, A, la, IW, liw, W, maxfrt, RHS, IW1, 
     *                   nsteps, ICNTL, INFO )

      USE GALAHAD_KINDS_double

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, la, liw, maxfrt, nsteps
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( liw ) :: IW
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( nsteps ) :: IW1
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 20 ) :: INFO
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( la ) :: A
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( maxfrt ) :: W
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: RHS

!  Dummy subroutine available with GALAHAD

      IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) > 0 ) 
     *  WRITE ( ICNTL( 2 ), 2000 )
      INFO( 1 ) = - 26
      RETURN

!  Non-executable statements

 2000 FORMAT( /, 
     *     ' We regret that the solution options that you have ', /, 
     *     ' chosen are not all freely available with GALAHAD.', //,
     *     ' If you have HSL (formerly the Harwell Subroutine',
     *     ' Library), this ', /,
     *     ' option may be enabled by replacing the dummy ', /,      
     *     ' subroutine MA27CD with its HSL namesake ', /,        
     *     ' and dependencies. See ', /,
     *     '   $GALAHAD/src/makedefs/packages for details.', //,
     *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MA27CD

      END SUBROUTINE MA27CD

      SUBROUTINE MA27ID( ICNTL, CNTL )

      USE GALAHAD_KINDS_double

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 30 ) :: ICNTL
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( 5 ) :: CNTL

!  Dummy subroutine available with GALAHAD

      ICNTL( 1 ) = - 1
      ICNTL( 4 ) = - 1
!     WRITE ( 6, 2000 )
      RETURN

!  Non-executable statements

!2000 FORMAT( /, 
!    *     ' We regret that the solution options that you have ', /, 
!    *     ' chosen are not all freely available with GALAHAD.', //,
!    *     ' If you have HSL (formerly the Harwell Subroutine',
!    *     ' Library), this ', /,
!    *     ' option may be enabled by replacing the dummy ', /,      
!    *     ' subroutine MA27ID with its HSL namesake ', /,        
!    *     ' and dependencies. See ', /,
!    *     '   $GALAHAD/src/makedefs/packages for details.', //,
!    *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MA27ID

      END SUBROUTINE MA27ID

      SUBROUTINE MA27QD( n, A, la, IW, liw, W, maxfnt, RHS, IW2, 
     *                   nblk, latop, ICNTL )

      USE GALAHAD_KINDS_double

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, la, liw, maxfnt, nblk
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: latop
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( nblk ) :: IW2
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( liw ) :: IW
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 30 ) :: ICNTL
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( la ) :: A
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: RHS
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( maxfnt ) :: W

!  Dummy subroutine available with GALAHAD

      WRITE ( 6, 2000 )
      STOP

!  Non-executable statements

 2000 FORMAT( /, 
     *     ' We regret that the solution options that you have ', /, 
     *     ' chosen are not all freely available with GALAHAD.', //,
     *     ' If you have HSL (formerly the Harwell Subroutine',
     *     ' Library), this ', /,
     *     ' option may be enabled by replacing the dummy ', /,      
     *     ' subroutine MA27QD with its HSL namesake ', /,        
     *     ' and dependencies. See ', /,
     *     '   $GALAHAD/src/makedefs/packages for details.', //,
     *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MA27QD

      END SUBROUTINE MA27QD
