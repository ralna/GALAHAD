! THIS VERSION: GALAHAD 4.3 - 2024-01-06 AT 07:40 GMT.

      SUBROUTINE MC77I( ICNTL, CNTL )
      USE GALAHAD_KINDS
      INTEGER ( KIND = ip_ ) :: LICNTL, LCNTL
      PARAMETER ( LICNTL = 10, LCNTL = 10 )
      INTEGER ( KIND = ip_ ) :: ICNTL( LICNTL )
      REAL ( KIND = sp_ ) :: CNTL( LCNTL )
      ICNTL( 1 ) = - 1
!     WRITE( 6, 
!    & "( ' We regret that the solution options that you have', /,
!    &    ' chosen are not all freely available with GALAHAD.', /,
!    &    ' If you have HSL (formerly the Harwell Subroutine', /,
!    &    ' Library), this option may be enabled by replacing', /,
!    &    ' the dummy subroutine MC77I with its HSL namesake ', /,
!    &    ' and dependencies. See ', /,
!    &    '   $GALAHAD/src/makedefs/packages for details.' )" )
      END

      SUBROUTINE MC77A( job, m, n, nnz, JCST, IRN, A, IW, liw, DW, ldw, 
     &                  ICNTL, CNTL, INFO, RINFO )
      USE GALAHAD_KINDS
      USE GALAHAD_SYMBOLS
      INTEGER ( KIND = ip_ ) :: licntl, lcntl, linfo, lrinfo
      PARAMETER ( licntl = 10, lcntl = 10, linfo = 10, lrinfo = 10 )
      INTEGER ( KIND = ip_ ) :: ICNTL( licntl ), INFO( linfo )
      REAL ( KIND = sp_ ) :: CNTL( lcntl ), RINFO( lrinfo )
      INTEGER ( KIND = ip_ ) :: job, m, n, nnz, liw, ldw
      INTEGER ( KIND = ip_ ) :: JCST( n + 1 ), IRN( nnz ), IW( liw )
      REAL ( KIND = sp_ ) :: A( nnz ), DW( ldw )
      IF ( ICNTL( 1 ) >= 0 ) WRITE( ICNTL( 1 ),
     & "( ' We regret that the solution options that you have', /,
     &    ' chosen are not all freely available with GALAHAD.', /,
     &    ' If you have HSL (formerly the Harwell Subroutine', /,
     &    ' Library), this option may be enabled by replacing', /,
     &    ' the dummy subroutine MC77A with its HSL namesake ', /,
     &    ' and dependencies. See ', /,
     &    '   $GALAHAD/src/makedefs/packages for details.' )" )
       INFO( 1 ) = GALAHAD_unavailable_option
      END

      SUBROUTINE MC77B( job, m, n, nnz, IRN, JCN, A, IW, liw, DW, ldw, 
     &                  ICNTL, CNTL, INFO, RINFO )
      USE GALAHAD_KINDS
      USE GALAHAD_SYMBOLS
      INTEGER ( KIND = ip_ ) :: licntl, lcntl, linfo, lrinfo
      PARAMETER ( licntl = 10, lcntl = 10, linfo = 10, lrinfo = 10 )
      INTEGER ( KIND = ip_ ) :: ICNTL( licntl ), INFO( linfo )
      REAL ( KIND = sp_ ) :: CNTL( lcntl ), RINFO( lrinfo )
      INTEGER ( KIND = ip_ ) :: job, m, n, nnz, liw, ldw
      INTEGER ( KIND = ip_ ) :: JCN( nnz ), IRN( nnz ), IW( liw )
      REAL ( KIND = sp_ ) :: A( nnz ), DW( ldw )
      IF ( ICNTL( 1 ) >= 0 ) WRITE( ICNTL( 1 ),
     & "( ' We regret that the solution options that you have', /,
     &    ' chosen are not all freely available with GALAHAD.', /,
     &    ' If you have HSL (formerly the Harwell Subroutine', /,
     &    ' Library), this option may be enabled by replacing', /,
     &    ' the dummy subroutine MC77B with its HSL namesake ', /,
     &    ' and dependencies. See ', /,
     &    '   $GALAHAD/src/makedefs/packages for details.' )" )
       INFO( 1 ) = GALAHAD_unavailable_option
      END

      SUBROUTINE MC77C( job, m, n, A, lda, IW, liw, DW, ldw, 
     &                  ICNTL, CNTL, INFO, RINFO )
      USE GALAHAD_KINDS
      USE GALAHAD_SYMBOLS
      INTEGER ( KIND = ip_ ) :: licntl, lcntl, linfo, lrinfo
      PARAMETER ( licntl = 10, lcntl = 10, linfo = 10, lrinfo = 10 )
      INTEGER ( KIND = ip_ ) :: ICNTL( licntl ), INFO( linfo )
      REAL ( KIND = sp_ ) :: CNTL( lcntl ), RINFO( lrinfo )
      INTEGER ( KIND = ip_ ) :: job, m, n, lda, liw, ldw
      INTEGER ( KIND = ip_ ) :: IW( liw )
      REAL ( KIND = sp_ ) :: A( lda, *), DW( ldw )
      IF ( ICNTL( 1 ) >= 0 ) WRITE( ICNTL( 1 ),
     & "( ' We regret that the solution options that you have', /,
     &    ' chosen are not all freely available with GALAHAD.', /,
     &    ' If you have HSL (formerly the Harwell Subroutine', /,
     &    ' Library), this option may be enabled by replacing', /,
     &    ' the dummy subroutine MC77C with its HSL namesake ', /,
     &    ' and dependencies. See ', /,
     &    '   $GALAHAD/src/makedefs/packages for details.' )" )
       INFO( 1 ) = GALAHAD_unavailable_option
      END
