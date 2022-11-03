! THIS VERSION: 2022-10-17 AT 08:45 GMT.
! Updated 14/05/2002: arguments for MA27 subroutines updated
! Updated 25/06/2002: additional warning information added
! Updated 14/03/2003: Warning removed from MA27ID

!-*-*-*-*-*-*-  G A L A H A D  -  MA27  S U B R O U T I N E S *-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  February 13th 1995

      SUBROUTINE MA27AD( n, nz, IRN, ICN, IW, liw, IKEEP, IW1, 
     *                   nsteps, iflag, ICNTL, CNTL, INFO, ops )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n, nz, liw
      INTEGER, INTENT( INOUT ) :: iflag
      INTEGER, INTENT( OUT ) :: nsteps
      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ) :: ops
      INTEGER, INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
      INTEGER, INTENT( OUT ), DIMENSION( liw ) :: IW
      INTEGER, INTENT( INOUT ), DIMENSION( n, 3 ) :: IKEEP
      INTEGER, INTENT( OUT ), DIMENSION( n, 2 ) :: IW1
      INTEGER, INTENT( IN ), DIMENSION( 30 ) :: ICNTL
      INTEGER, INTENT( OUT ), DIMENSION( 20 ) :: INFO
      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ), 
     *                                DIMENSION( 5 ) :: CNTL

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

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n, nz, la, liw, nsteps
      INTEGER, INTENT( OUT ) :: maxfrt
      INTEGER, INTENT( IN ), DIMENSION( nz ) :: IRN, ICN
      INTEGER, INTENT( OUT ), DIMENSION( liw ) :: IW
      INTEGER, INTENT( IN ), DIMENSION( n, 3 ) :: IKEEP
      INTEGER, INTENT( OUT ), DIMENSION( n ) :: IW1
      INTEGER, INTENT( IN ), DIMENSION( 30 ) :: ICNTL
      INTEGER, INTENT( OUT ), DIMENSION( 20 ) :: INFO
      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ), 
     *                                DIMENSION( 5 ) :: CNTL
      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ), 
     *                                DIMENSION( la ) :: A

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

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n, la, liw, maxfrt, nsteps
      INTEGER, INTENT( IN ), DIMENSION( liw ) :: IW
      INTEGER, INTENT( OUT ), DIMENSION( nsteps ) :: IW1
      INTEGER, INTENT( IN ), DIMENSION( 30 ) :: ICNTL
      INTEGER, INTENT( OUT ), DIMENSION( 20 ) :: INFO
      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ), 
     *                                DIMENSION( la ) :: A
      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), 
     *                                DIMENSION( maxfrt ) :: W
      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ), 
     *                                DIMENSION( n ) :: RHS

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

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( OUT ), DIMENSION( 30 ) :: ICNTL
      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), 
     *                                DIMENSION( 5 ) :: CNTL

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
      INTEGER, INTENT( IN ) :: n, la, liw, maxfnt, nblk
      INTEGER, INTENT( OUT ) :: latop
      INTEGER, INTENT( IN ), DIMENSION( nblk ) :: IW2
      INTEGER, INTENT( OUT ), DIMENSION( liw ) :: IW
      INTEGER, INTENT( IN ), DIMENSION( 30 ) :: ICNTL
      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ), 
     *                                DIMENSION( la ) :: A
      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ), 
     *                                DIMENSION( n ) :: RHS
      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), 
     *                                DIMENSION( maxfnt ) :: W

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
