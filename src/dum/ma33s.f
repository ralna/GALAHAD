! THIS VERSION: GALAHAD 4.3 - 2024-01-06 AT 07:50 GMT.

!-*-*-*-*-*-  G A L A H A D  -  D U M M Y   M A 3 3  S U B R O U T I N E *-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  April 10th 2006

      SUBROUTINE MA33A( n, ICN, A, licn, LENR, LENRL, IDISP, IP, IQ,
     *                  IRN, lirn, LENC, IFIRST, LASTR, NEXTR, LASTC,
     *                  NEXTC, IPTR, IPC, u, iflag, ICNTL, CNTL, INFO,
     *                  RINFO )

      USE GALAHAD_KINDS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, licn, lirn
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: iflag
      REAL ( KIND = sp_ ), INTENT( INOUT ) :: u
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( licn ) ::  ICN
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: LENR
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) ::  LENRL
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( 2 ) :: IDISP
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: IP, IQ
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( lirn ) :: IRN
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) ::  IPC
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) ::  IPTR
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) ::  LENC
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) ::  IFIRST
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) ::  LASTR
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) ::  NEXTR
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) ::  LASTC
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) ::  NEXTC
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 10 ) :: INFO
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 10 ) :: ICNTL
      REAL ( KIND = sp_ ), INTENT( INOUT ), DIMENSION( licn ) :: A
      REAL ( KIND = sp_ ), INTENT( OUT ), DIMENSION( 5 ) :: RINFO
      REAL ( KIND = sp_ ), INTENT( IN ), DIMENSION( 5 ) :: CNTL

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
     *     ' subroutine MA33A with its HSL namesake ', /,
     *     ' and dependencies. See ', /,
     *     '   $GALAHAD/src/makedefs/packages for details.', //,
     *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MA33A

      END SUBROUTINE MA33A


      SUBROUTINE MA33C( n, ICN, A, licn, LENR, LENRL, LENOFF, IDISP,
     *                  IP, IQ, X, W, mtype, RINFO )

      USE GALAHAD_KINDS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, licn, mtype
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( licn ) :: ICN
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n ) :: LENR
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n ) :: LENRL
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n ) :: LENOFF
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 2 ) :: IDISP
       INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: IP, IQ
       REAL ( KIND = sp_ ), INTENT( IN ), DIMENSION( licn ) :: A
       REAL ( KIND = sp_ ), INTENT( OUT ), DIMENSION( n ) :: W
       REAL ( KIND = sp_ ), INTENT( INOUT ), DIMENSION( n ) :: X
       REAL ( KIND = sp_ ), INTENT( INOUT ), DIMENSION( 5 ) :: RINFO

!  Dummy subroutine available with GALAHAD

!     WRITE ( 6, 2000 )
      STOP

!  Non-executable statements

!2000 FORMAT( /,
!    *     ' We regret that the solution options that you have ', /,
!    *     ' chosen are not all freely available with GALAHAD.', //,
!    *     ' If you have HSL (formerly the Harwell Subroutine',
!    *     ' Library), this ', /,
!    *     ' option may be enabled by replacing the dummy ', /,
!    *     ' subroutine MA33C with its HSL namesake ', /,
!    *     ' and dependencies. See ', /,
!    *     '   $GALAHAD/src/makedefs/packages for details.', //,
!    *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MA33C

      END SUBROUTINE MA33C

      SUBROUTINE MA33I( ICNTL, CNTL )

      USE GALAHAD_KINDS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( 10 ) :: ICNTL
      REAL ( KIND = KIND( 1.0e+0 ) ), INTENT( OUT ),
     *                                DIMENSION( 5 ) :: CNTL

!  Dummy subroutine available with GALAHAD

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
!    *     ' subroutine MA33I with its HSL namesake ', /,
!    *     ' and dependencies. See ', /,
!    *     '   $GALAHAD/src/makedefs/packages for details.', //,
!    *     ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine MA33I

      END SUBROUTINE MA33I
