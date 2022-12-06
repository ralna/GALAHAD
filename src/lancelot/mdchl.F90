! THIS VERSION: GALAHAD 2.4 - 06/04/2010 AT 08:00 GMT.

!-*-*-*-*-*-*-*-  L A N C E L O T  -B-  MDCHL   M O D U L E  *-*-*-*-*-*-*-*

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  February 3rd 1995

   MODULE LANCELOT_MDCHL_double

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: MDCHL_mcfa, MDCHL_syprc, MDCHL_gmps, MDCHL_iccga, MDCHL_iccgb,  &
               MDCHL_block_type, MDCHL_get_singular_direction, MDCHL_get_donc

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PRIVATE, PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PRIVATE, PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PRIVATE, PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PRIVATE, PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PRIVATE, PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PRIVATE, PARAMETER :: cmax = ten ** 20

   CONTAINS

!-*-*-*-  L A N C E L O T  -B-  MDCHL_mcfa S U B R O U T I N E -*-*-*-*

     SUBROUTINE MDCHL_mcfa( n, nz, IRN, ICN, A, la, IW, liw, IKEEP,            &
                            nsteps, maxfrt, IW1, ICNTL, CNTL, INFO, DIAG )
!                           OMEGA )

!  Given an elimination ordering, factorize a symmetric matrix A

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER n, nz, la, liw, nsteps, maxfrt
     INTEGER, DIMENSION( * ) :: IRN, ICN
     INTEGER, DIMENSION( liw ) :: IW
     INTEGER, DIMENSION( n, 3 ) :: IKEEP
     INTEGER, DIMENSION( n ) :: IW1
     INTEGER, DIMENSION( 30 ) :: ICNTL
     INTEGER, DIMENSION( 20 ) :: INFO
     REAL ( KIND = wp ), DIMENSION( 5 ) :: CNTL
     REAL ( KIND = wp ), DIMENSION( la ) :: A
     REAL ( KIND = wp ), DIMENSION( n ) :: DIAG
!    REAL ( KIND = wp ), DIMENSION( n ) :: OMEGA

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: k, kz, nz1, iphase, j2, j1, irows
     INTEGER :: len, nrows, ipos, kblk, iapos, ncols, iblk
     REAL ( KIND = wp ) :: addon
     
     INFO(1) = 0
     IF ( ICNTL( 3 ) > 0 .AND. ICNTL( 2 ) > 0 ) THEN

!  Print input parameters

       WRITE( ICNTL( 2 ), 2010 ) n, nz, la, liw, nsteps, CNTL( 1 )
       kz = MIN( 6, nz ) ; IF ( ICNTL( 3 ) > 1 ) kz = nz
       IF ( nz > 0 ) WRITE( ICNTL( 2 ), 2020 )                                 &
         ( A( k ), IRN( k ), ICN( k ), k = 1, kz )
       k = MIN( 9, n ) ; IF ( ICNTL( 3 ) > 1 ) k = n
       IF ( k > 0 ) WRITE( ICNTL( 2 ), 2030 ) IKEEP( : k, 1 )
       k = MIN( k, nsteps )
       IF ( k > 0 ) THEN
         WRITE( ICNTL( 2 ), 2040 ) IKEEP( : k, 2 )
         WRITE( ICNTL( 2 ), 2050 ) IKEEP( : k, 3 )
       END IF
     END IF
     IF ( n >= 1 .AND. n <= ICNTL( 4 ) ) THEN
       IF ( nz < 0 ) THEN
         INFO( 1 ) = - 2
         IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2080 ) INFO( 1 )
         IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2110 ) nz
         GO TO 130
       END IF
       IF ( liw < nz ) THEN
         INFO( 1 ) = - 3 ; INFO( 2 ) = nz
         GO TO 130
       END IF
       IF ( la < nz + n ) THEN
         INFO( 1 ) = - 4 ; INFO( 2 ) = nz + n
         GO TO 130
       END IF

!  Set phase of Cholesky modification

       iphase = 1

!  Sort

       CALL MDCHL_mcfb( n, nz, nz1, A, la, IRN, ICN, IW, liw, IKEEP,           &
                        IW1, ICNTL, INFO, DIAG, addon )
       IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) >= 2 )                             &
         WRITE( ICNTL( 2 ), 2000 ) addon 
       IF ( INFO( 1 ) == - 3 .OR. INFO( 1 ) == - 4 ) GO TO 130

!  Factorize

       CALL MDCHL_mcfc( n, nz1, A, la, IW, liw, IKEEP, IKEEP( 1, 3 ),          &
                        nsteps, maxfrt, IKEEP( 1, 2 ), IW1,                    &
                        ICNTL, CNTL, INFO, DIAG, addon, iphase )
!                       CNTL, INFO, DIAG, OMEGA, addon, iphase )
       IF ( INFO( 1 ) == - 3 .OR. INFO( 1 ) == - 4 ) GO TO 130
       IF ( INFO( 1 ) == - 5 ) THEN
         IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2080 ) INFO( 1 )
         IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2190 ) INFO( 2 )
       END IF
       IF ( INFO( 1 ) == - 6 ) THEN
         IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2080 ) INFO( 1 )
         IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2210 )
       END IF

! **** Warning message ****

       IF ( INFO( 1 ) == 3 .AND. ICNTL( 2 ) > 0 )                              &
         WRITE( ICNTL( 2 ), 2060 ) INFO( 1 ), INFO( 2 )

! **** Error returns ****

     ELSE
       INFO( 1 ) = - 1
       IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2080 ) INFO( 1 )
       IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2090 ) n
     END IF

 130 CONTINUE
     IF ( INFO( 1 ) == - 3 ) THEN
       IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2080 ) INFO( 1 )
       IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2140 ) liw, INFO( 2 )
     ELSE IF ( INFO( 1 ) == - 4 ) THEN
       IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2080 ) INFO( 1 )
       IF ( ICNTL( 1 ) > 0 ) WRITE( ICNTL( 1 ), 2170 ) la, INFO( 2 )
     END IF
     IF ( ICNTL( 3 ) > 0 .AND. ICNTL( 2 ) > 0 ) THEN

!  Print output parameters

       WRITE( ICNTL( 2 ), 2230 ) maxfrt, INFO( 1 ),INFO( 9 ), INFO( 10 ), &
          INFO( 12 ), INFO( 13 ),INFO( 14 ), INFO( 2 ) 
       IF ( INFO( 1 ) >= 0 ) THEN

!  Print out matrix factors from MCFA/B

         kblk = ABS( IW( 1 ) + 0 )
         IF ( kblk /= 0 ) THEN
           IF ( ICNTL( 3 ) == 1 ) kblk = 1
           ipos = 2 ; iapos = 1
           DO iblk = 1, kblk
             ncols = IW( ipos ) ; nrows = IW( ipos + 1 )
             j1 = ipos + 2
             IF ( ncols <= 0 ) THEN
               ncols = - ncols ; nrows = 1
               j1 = j1 - 1
             END IF
             WRITE( ICNTL( 2 ), 2250 ) iblk, nrows, ncols
             j2 = j1 + ncols - 1
             ipos = j2 + 1
             WRITE( ICNTL( 2 ), 2260 ) IW( j1 : j2 )
             WRITE( ICNTL( 2 ), 2270 )
             len = ncols
             DO irows = 1, nrows
               j1 = iapos ; j2 = iapos + len - 1
               WRITE( ICNTL( 2 ), 2280 ) A( j1 : j2 )
               len = len - 1
               iapos = j2 + 1
             END DO
           END DO
         END IF
       END IF
     END IF

     RETURN

!  Non executable statements

 2000 FORMAT( ' addon = ', ES12.4 )
 2010 FORMAT( //, ' entering MCFA  with      n     nz     la    liw ',         &
              ' nsteps      u', / , 21X, 5I7, F7.2 )
 2020 FORMAT( ' Matrix non-zeros', 2( ES16.4, 2I6 ), /,                        &
              ( 17X, ES16.4, 2I6, ES16.4, 2I6 ) )
 2030 FORMAT( ' IKEEP( ., 1 )=', 10I6, /, ( 12X, 10I6 ) )
 2040 FORMAT( ' IKEEP( ., 2 )=', 10I6, /, ( 12X, 10I6 ) )
 2050 FORMAT( ' IKEEP( ., 3 )=', 10I6, /, ( 12X, 10I6 ) )
 2060 FORMAT( ' *** Warning message from subroutine MCFA *** info(1) =',       &
              I2, /, 5X, 'matrix is singular. Rank=', I5 )
 2080 FORMAT( ' **** Error return from MCFA **** info(1) =', I3 )
 2090 FORMAT( ' Value of n out of range ... =', I10 )
 2110 FORMAT( ' Value of nz out of range .. =', I10 )
 2140 FORMAT( ' liw too small, must be increased from', I10,                   &
              ' to at least', I10 )
 2170 FORMAT( ' la too small, must be increased from ', I10,                   &
              ' to at least', I10 )
 2190 FORMAT( ' Zero pivot at stage', I10,                                     &
              ' when input matrix declared definite' )
 2210 FORMAT( ' Change in sign of pivot encountered ',                         &
              ' when factoring allegedly definite matrix' )
 2230 FORMAT( /' Leaving MCFA with maxfrt  info(1) nrlbdu nirbdu ncmpbr',      &
               ' ncmpbi   ntwo ierror', /, 20X, 8I7 )
 2250 FORMAT( ' Block pivot =', I8,' nrows =', I8,' ncols =', I8 )
 2260 FORMAT( ' Column indices =', 10I6, /, ( 17X, 10I6 ) )
 2270 FORMAT( ' Real entries .. each row starts on a new line' )
 2280 FORMAT( 5ES16.8 )

!  End of subroutine MDCHL_mcfa

     END SUBROUTINE MDCHL_mcfa

!-*-*-*-  L A N C E L O T  -B-  MDCHL_mcfb S U B R O U T I N E -*-*-*-*

     SUBROUTINE MDCHL_mcfb( n, nz, nz1, A, la, IRN, ICN, IW, liw, PERM,        &
                            IW2, ICNTL, INFO, DIAG, addon )

!  Sort the entries of A prior to the factorization

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER n, nz, nz1, la, liw
     REAL ( KIND = wp ) addon
     INTEGER, DIMENSION( * ) :: IRN, ICN
     INTEGER, DIMENSION( liw ) :: IW
     INTEGER, DIMENSION( n ) :: PERM, IW2
     INTEGER, DIMENSION( 30 ) :: ICNTL
     INTEGER, DIMENSION( 20 ) :: INFO
     REAL ( KIND = wp ), DIMENSION( la ) :: A
     REAL ( KIND = wp ), DIMENSION( n ) :: DIAG

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: k, iold, inew, jold, ia, jnew, j2, j1, iiw, jj, ii
     INTEGER :: ich, i, ipos, jpos
     REAL ( KIND = wp ) :: anext, anow, machep, maxdag

! ** Obtain machep

     machep = EPSILON( one )
     INFO( 1 ) = 0

!  Initialize work array (IW2) in preparation for counting numbers of 
!  non-zeros in the rows and initialize last n entries in A which will
!  hold the diagonal entries

     ia = la
     IW2 = 1
     A( ia - n + 1 : ia ) = zero

!  Scan input copying row indices from IRN to the first nz positions
!  in IW. The negative of the index is held to flag entries for
!  the in-place sort. Entries in IW corresponding to diagonals and
!  entries with out-of-range indices are set to zero. For diagonal entries, 
!  reals are accumulated in the last n locations OF A.
!  The number of entries in each row of the permuted matrix is
!  accumulated in IW2. Indices out of range are ignored after being counted 
!  and after appropriate messages have been printed.

     INFO( 2 ) = 0

!  nz1 is the number of non-zeros held after indices out of range have
!  been ignored and diagonal entries accumulated.

     nz1 = n
     IF ( nz /= 0 ) THEN
       DO k = 1, nz
         iold = IRN( k )
         IF ( iold <= n .AND. iold > 0 ) THEN
           jold = ICN( k )
           IF ( jold <= n .AND. jold > 0 ) THEN
             inew = PERM( iold )
             jnew = PERM( jold )
             IF ( inew == jnew ) THEN
               ia = la - n + iold
               A( ia ) = A( ia ) + A( k )
               IW( k ) = 0
               CYCLE
             END IF
             inew = MIN( inew, jnew )

!  Increment number of entries in row inew.

             IW2( inew ) = IW2( inew ) + 1
             IW( k ) = - iold
             nz1 = nz1 + 1
             CYCLE

!  Entry out of range. It will be ignored and a flag set.

           END IF
         END IF
         INFO( 1 ) = 1
         INFO( 2 ) = INFO( 2 ) + 1
         IF ( INFO( 2 ) <= 1 .AND. ICNTL( 2 ) > 0 )                            &
           WRITE( ICNTL( 2 ), 2040 ) INFO( 1 )
         IF ( INFO( 2 ) <= 10 .AND. ICNTL( 2 ) > 0 )                           &
           WRITE( ICNTL( 2 ), 2050 ) k, IRN( k ), ICN( k )
         IW( k ) = 0
       END DO

!  Calculate pointers (in IW2) to the position immediately after the end
!  of each row.

     END IF

!  Room is included for the diagonals.

     IF ( nz >= nz1 .OR. nz1 == n ) THEN
       k = 1
       DO i = 1, n
         k = k + IW2( i )
         IW2( i ) = k
       END DO
     ELSE

!  Room is not included for the diagonals.

       k = 1
       DO i = 1, n
         k = k + IW2( i ) - 1
         IW2( i ) = k
       END DO

!  Fail if insufficient space in arrays A or IW.

     END IF

!  **** Error return ****

     IF ( nz1 > liw ) THEN
       INFO( 1 ) = - 3 ; INFO( 2 ) = nz1 ; RETURN
     END IF
     
     IF ( nz1 + n > la ) THEN
       INFO( 1 ) = - 4 ; INFO( 2 ) = nz1 + n ; RETURN
     END IF 

!  Now run through non-zeros in order placing them in their new
!  position and decrementing appropriate IW2 entry. If we are
!  about to overwrite an entry not yet moved, we must deal with
!  this at this time.

     IF ( nz1 /= n ) THEN
 L140: DO k = 1, nz
         iold = - IW( k )
         IF ( iold <= 0 ) CYCLE  L140
         jold = ICN( k )
         anow = A( k )
         IW( k ) = 0
         DO ich = 1, nz
           inew = PERM( iold ) ; jnew = PERM( jold )
           inew = MIN( inew, jnew )
           IF ( inew == PERM( jold ) ) jold = iold
           jpos = IW2( inew ) - 1
           iold = -IW( jpos )
           anext = A( jpos )
           A( jpos ) = anow
           IW( jpos ) = jold
           IW2( inew ) = jpos
           IF ( iold == 0 ) CYCLE  L140
           anow = anext
           jold = ICN( jpos )
         END DO
       END DO L140
       IF ( nz < nz1 ) THEN

!  Move up entries to allow for diagonals.

         ipos = nz1 ; jpos = nz1 - n
         DO ii = 1, n
           i = n - ii + 1
           j1 = IW2( i ) ; j2 = jpos
           IF ( j1 <= jpos ) THEN
             DO jj = j1, j2
               IW( ipos ) = IW( jpos )
               A( ipos ) = A( jpos )
               ipos = ipos - 1
               jpos = jpos - 1
             END DO
           END IF
           IW2( i ) = ipos + 1
           ipos = ipos - 1
         END DO

!  Run through rows inserting diagonal entries and flagging beginning
!  of each row by negating first column index.

       END IF
     END IF
     maxdag = machep
     DO iold = 1, n
       inew = PERM( iold )
       jpos = IW2( inew ) - 1
       ia = la - n + iold
       A( jpos ) = A( ia )

!  Set diag to value of diagonal entry (original numbering)

       DIAG( iold ) = A( ia )
       maxdag = MAX( maxdag, ABS( A( ia ) ) )
       IW( jpos ) = - iold
     END DO

!  Compute addition to off-diagonal 1-norm

     addon = maxdag * machep ** 0.75

!  Move sorted matrix to the end of the arrays

     ipos = nz1
     ia = la
     iiw = liw
     DO i = 1, nz1
        A( ia ) = A( ipos )
        IW( iiw ) = IW( ipos )
        ipos = ipos - 1
        ia = ia - 1
        iiw = iiw - 1
     END DO
     
     RETURN

!  Non executable statements

 2040 FORMAT( ' *** Warning message from subroutine MCFB *** iflag =', I2 )
 2050 FORMAT( I6, 'th non-zero (in row', I6, ' and column ', I6, ') ignored' )

!  End of subroutine MDCHL_mcfb

     END SUBROUTINE MDCHL_mcfb

!-*-*-*-  L A N C E L O T  -B-  MDCHL_mcfc S U B R O U T I N E -*-*-*-*

     SUBROUTINE MDCHL_mcfc( n, nz, A, la, IW, liw, PERM, NSTK, nsteps,         &
                            maxfrt, NELIM, IW2, ICNTL, CNTL, INFO, DIAG,       &
                            addon, iphase )
!                           OMEGA, addon, iphase )

!  Perform the multifrontal factorization

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER n, nz, la, liw, nsteps, maxfrt, iphase
     REAL ( KIND = wp ) addon
     INTEGER, DIMENSION( liw ) :: IW
     INTEGER, DIMENSION( n ) :: PERM
     INTEGER, DIMENSION( nsteps ) :: NSTK, NELIM
     INTEGER, DIMENSION( n ) :: IW2
     INTEGER, DIMENSION( 30 ) :: ICNTL
     INTEGER, DIMENSION( 20 ) :: INFO
     REAL ( KIND = wp ), DIMENSION( 5 ) :: CNTL
     REAL ( KIND = wp ), DIMENSION( la ) :: A
     REAL ( KIND = wp ), DIMENSION( n ) :: DIAG
!    REAL ( KIND = wp ), DIMENSION( n ) :: OMEGA

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: idummy, numorg, jnew, jj, j, laell, lapos2, ifr, iorg
     INTEGER :: jdummy, j2, iell, jcol, npiv, newel, istk, i, azero
     INTEGER :: ltopst, lnass, numstk, jfirst, nfront, jlast, j1, jnext
     INTEGER :: iswap, ibeg, iexch, krow, ipos, liell, kmax, ioldps, iend
     INTEGER :: kdummy, lnpiv, irow, jjj, jay, kk, ipiv, npivp1, jpiv
     INTEGER :: istk2, iwpos, k, nblk, iass, nass, numass, iinput, ntotpv
     INTEGER :: posfac, astk, astk2, apos, apos1, apos2, ainput, pivsiz
     INTEGER :: ntwo, neig, ncmpbi, ncmpbr, nrlbdu, nirbdu
     REAL ( KIND = wp ) :: amax, rmax, swap, amult, w1, onenrm, uu
     
!    LOGICAL, DIMENSION( n ) :: PIVOT_set
     
!    PIVOT_set = .FALSE.
     
     IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) >= 2 ) THEN
        WRITE( ICNTL( 2 ), 2000 ) DIAG( : MIN( n, 4 ) )
        WRITE( ICNTL( 2 ), 2100 ) iphase
     END IF

!  Initialization.
!  nblk is the number of block pivots used.

     nblk = 0 ; ntwo = 0 ; neig = 0
     ncmpbi = 0 ; ncmpbr = 0 ; maxfrt = 0
     nrlbdu = 0 ; nirbdu = 0

!  A private variable uu is set to u, so that u will remain unaltered.

     uu = MIN( CNTL( 1 ), half )
     uu = MAX( uu, - half )

! OMEGA is an array of adjustments to the diagonal

     IW2 = 0
!    OMEGA = 0.0

!  iwpos is pointer to first free position for factors in IW.
!  posfac is pointer for factors in A. At each pass through the
!      major loop posfac initially points to the first free location
!      in A and then is set to the position of the current pivot in A.
!  istk is pointer to top of stack in IW.
!  istk2 is pointer to bottom of stack in IW (needed by compress).
!  astk is pointer to top of stack in A.
!  astk2 is pointer to bottom of stack in A (needed by compress).
!  iinput is pointer to current position in original rows in IW.
!  ainput is pointer to current position in original rows in A.
!  azero is pointer to last position zeroed in A.
!  ntotpv is the total number of pivots selected. This is used
!      to determine whether the matrix is singular.

         iwpos = 2
         posfac = 1
         istk = liw - nz + 1 ; istk2 = istk - 1
         astk = la - nz + 1 ; astk2 = astk - 1
         iinput = istk ; ainput = astk
         azero = 0
         ntotpv = 0

!  numass is the accumulated number of rows assembled so far.

         numass = 0

!  Each pass through this main loop performs all the operations
!      associated with one set of assembly/eliminations.

         DO iass = 1, nsteps

!  nass will be set to the number of fully assembled variables in
!      current newly created element.

            nass = NELIM( iass )

!  newel is a pointer into IW to control output of integer information
!      for newly created element.

            newel = iwpos + 1

!  Symbolically assemble incoming rows and generated stack elements
!  ordering the resultant element according to permutation PERM.  We
!  assemble the stack elements first because these will already be ordered.

!  Set header pointer for merge of index lists.

            jfirst = n + 1

!  Initialize number of variables in current front.

            nfront = 0
            numstk = NSTK( iass )
            ltopst = 1
            lnass = 0

!  Jump if no stack elements are being assembled at this stage.

            IF ( numstk /= 0 ) THEN
               j2 = istk - 1
               lnass = nass
               ltopst = ( ( IW( istk ) + 1 ) * IW( istk ) ) / 2
               DO iell = 1, numstk

!  Assemble element iell placing the indices into a linked list in IW2 
!  ordered according to PERM.

                  jnext = jfirst
                  jlast = n + 1
                  j1 = j2 + 2
                  j2 = j1 - 1 + IW( j1 - 1 )

!  Run through index list of stack element iell.

                  DO jj = j1, j2
                     j = IW( jj )
                     IF ( IW2( j ) > 0 ) CYCLE
                     jnew = PERM( j )

!  If variable was previously fully summed but was not pivoted on earlier 
!  because of numerical test, increment number of fully summed rows/columns 
!  in front.

                     IF ( jnew <= numass ) nass = nass + 1

!  Find position in linked list for new variable.  Note that we start
!  from where we left off after assembly of previous variable.

                     DO idummy = 1, n
                        IF ( jnext == n + 1 ) EXIT
                        IF ( PERM( jnext ) > jnew ) EXIT
                        jlast = jnext
                        jnext = IW2( jlast )
                     END DO

                     IF ( jlast == n + 1 ) THEN
                        jfirst = j
                     ELSE
                        IW2( jlast ) = j
                     END IF

                     IW2( j ) = jnext
                     jlast = j

!  Increment number of variables in the front.

                     nfront = nfront + 1
                  END DO
               END DO
               lnass = nass - lnass
            END IF

!  Now incorporate original rows.  Note that the columns in these rows need not
!  be in order. We also perform a swap so that the diagonal entry is the first
!  in its row. This allows us to avoid storing the inverse of array PERM.

            numorg = NELIM( iass )
            j1 = iinput
 L150:      DO iorg = 1, numorg
               j = - IW( j1 )
               DO idummy = 1, liw
                  jnew = PERM( j )

!  Jump if variable already included.

                  IF ( IW2( j ) <= 0 ) THEN

!  Here we must always start our search at the beginning.

                     jlast = n + 1
                     jnext = jfirst
                     DO jdummy = 1, n
                        IF ( jnext == n + 1 ) EXIT
                        IF ( PERM( jnext ) > jnew ) EXIT
                        jlast = jnext
                        jnext = IW2( jlast )
                     END DO
                     IF ( jlast == n + 1 ) THEN
                        jfirst = j
                     ELSE
                        IW2( jlast ) = j
                     END IF
                     IW2( j ) = jnext

!  Increment number of variables in front.

                     nfront = nfront + 1
                  END IF

                  j1 = j1 + 1
                  IF ( j1 > liw ) CYCLE L150
                  j = IW( j1 )
                  IF ( j < 0 ) CYCLE L150
               END DO
            END DO L150

!  Now run through linked list IW2 putting indices of variables in new
!  element into IW and setting IW2 entry to point to the relative
!  position of the variable in the new element.

            IF ( newel + nfront >= istk ) THEN

!  Compress IW.

               CALL MDCHL_compress( A, IW, istk, istk2, iinput, 2,             &
                                    ncmpbr, ncmpbi )
               IF ( newel + nfront >= istk ) THEN
                  INFO( 2 ) = liw + 1 + newel + nfront - istk
                  INFO( 1 ) = - 3
                  RETURN
               END IF
            END IF

            j = jfirst
            DO ifr = 1, nfront
               newel = newel + 1
               IW( newel ) = j
               jnext = IW2( j )
               IW2( j ) = newel - iwpos - 1
               j = jnext
            END DO

!  Assemble reals into frontal matrix.

            maxfrt = MAX( maxfrt, nfront )
            IW( iwpos ) = nfront

!  First zero out frontal matrix as appropriate first checking to see
!  if there is sufficient space.

            laell = ( ( nfront + 1 ) * nfront )/2
            apos2 = posfac + laell - 1
            IF ( numstk /= 0 ) lnass = lnass * ( 2 * nfront - lnass + 1 ) / 2
            IF ( posfac + lnass - 1 < astk ) THEN
               IF ( apos2 < astk + ltopst - 1 ) GO TO 190
            END IF

!  Compress A.

            CALL MDCHL_compress( A, IW, astk, astk2, ainput, 1, ncmpbr, ncmpbi )

! Error returns

            IF ( posfac + lnass - 1 >= astk ) THEN
               INFO( 1 ) = - 4
               INFO( 2 ) = la + MAX( posfac + lnass, apos2 - ltopst + 2 ) - astk
               RETURN
            END IF
            IF ( apos2 >= astk + ltopst - 1 ) THEN
               INFO( 1 ) = - 4
               INFO( 2 ) = la + MAX( posfac + lnass, apos2 - ltopst + 2 ) - astk
               RETURN
            END IF

  190       CONTINUE
            IF ( apos2 > azero ) THEN
               apos = azero + 1
               lapos2 = MIN( apos2, astk - 1 )
               IF ( lapos2 >= apos ) THEN
                  A( apos : lapos2 ) = zero
               END IF
               azero = apos2
            END IF

!  Jump if there are no stack elements to assemble.

            IF ( numstk /= 0 ) THEN

!  Place reals corresponding to stack elements in correct positions in A.

               DO iell = 1, numstk
                  j1 = istk + 1 ; j2 = istk + IW( istk )
                  DO jj = j1, j2
                     irow = IW2( IW( jj ) )
                     apos = posfac + MDCHL_idiag( nfront, irow )
                     DO jjj = jj, j2
                        j = IW( jjj )
                        apos2 = apos + IW2( j ) - irow
                        A( apos2 ) = A( apos2 ) + A( astk )
                        A( astk ) = zero
                        astk = astk + 1
                     END DO
                  END DO

!  Increment stack pointer.

                  istk = j2 + 1
               END DO
            END IF

!  Incorporate reals from original rows.

 L280:      DO iorg = 1, numorg
               j = - IW( iinput )

!  We can do this because the diagonal is now the first entry.

               irow = IW2( j )
               apos = posfac + MDCHL_idiag( nfront, irow )

!  The following loop goes from 1 to nz because there may be duplicates.

               DO idummy = 1, nz
                  apos2 = apos + IW2( j ) - irow
                  A( apos2 ) = A( apos2 ) + A( ainput )
                  ainput = ainput + 1 ; iinput = iinput + 1
                  IF ( iinput > liw ) CYCLE L280
                  j = IW( iinput )
                  IF ( j < 0 ) CYCLE L280
               END DO
            END DO L280

!  Reset IW2 and numass.

            numass = numass + numorg
            j1 = iwpos + 2 ; j2 = iwpos + nfront + 1
            IW2( IW( j1: j2 ) ) = 0

!  Perform pivoting on assembled element.
!  npiv is the number of pivots so far selected.
!  lnpiv is the number of pivots selected after the last pass through
!      the the following loop.

            lnpiv = - 1 ; npiv = 0

!           WRITE( 6, 2030 ) iwpos, IW( : nfront + 2 )

            DO kdummy = 1, nass
               IF ( npiv == nass ) EXIT
               IF ( npiv == lnpiv ) EXIT
               lnpiv = npiv ; npivp1 = npiv + 1

!  jpiv is used as a flag to indicate when 2 by 2 pivoting has occurred
!      so that ipiv is incremented correctly.

               jpiv = 1

!  nass is maximum possible number of pivots. We either take the diagonal 
!  entry or the 2 by 2 pivot with the largest off-diagonal at each stage.
!  Each pass through this loop tries to choose one pivot.

               DO ipiv = npivp1, nass
                  jpiv = jpiv - 1
                  IF ( jpiv == 1 ) CYCLE
                  apos = posfac + MDCHL_idiag( nfront - npiv, ipiv - npiv )

!  If the user has indicated that the matrix is definite, we do not need to 
!  test for stability but we do check to see if the pivot is non-zero or has 
!  changed sign. If it is zero, we exit with an error. If it has changed sign
!  and u was set negative, then we again exit immediately. If the pivot changes
!  sign and u was zero, we continue with the factorization but print a warning
!  message on unit mp.

!  isnpiv holds a flag for the sign of the pivots to date so that a sign 
!  change when decomposing an allegedly definite matrix can be detected.

!            IF ( uu > zero ) GO TO 320
!            IF ( A( apos ) == zero ) GO TO 790

!  Jump if this is not the first pivot to be selected.

!            IF ( ntotpv > 0 ) GO TO 300

!  Set isnpiv.

!            IF ( A( apos ) > zero ) isnpiv = 1
!            IF ( A( apos ) < zero ) isnpiv = - 1
!  300       IF ( A( apos ) > zero .AND. isnpiv == 1 ) GO TO 560
!            IF ( A( apos ) < zero .AND. isnpiv == - 1 ) GO TO 560
!            IF ( INFO( 1 ) /= 2 ) INFO( 2 ) = 0
!            INFO( 2 ) = INFO( 2 ) + 1
!            INFO( 1 ) = 2
!            i = ntotpv + 1
!            IF ( ICNTL( 2 ) > 0 .AND. INFO( 2 ) <= 10 )                       &
!               WRITE( ICNTL( 2 ), 2040 ) INFO( 1 ), i
!            isnpiv = - isnpiv
!            IF ( uu == zero ) GO TO 560
!            GO TO 800

!  First check if pivot is positive.

                  IF ( A( apos ) <= zero ) THEN
                    IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) >= 2 )                &
                      WRITE( ICNTL( 2 ), 2050 ) npiv
                    iphase = 2
                  END IF
                  amax = zero
                  rmax = amax

! i is pivot row.

                  i = IW( iwpos + ipiv + 1 )
                  IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) >= 2 ) THEN
                    WRITE( ICNTL( 2 ), 2060 ) npiv, i
                    WRITE( ICNTL( 2 ), 2010 ) A( apos )
                  END IF

!  Find largest entry to right of diagonal in row of prospective pivot
!  in the fully-summed part. Also record column of this largest entry.
!  onenrm is set to 1-norm of off-diagonals in row.

                  onenrm = 0.0
                  j1 = apos + 1 ; j2 = apos + nfront - ipiv
                  DO jj = j1, j2
                     jay = jj - j1 + 1
                     IF ( iphase == 1 ) THEN
                        j = IW( iwpos + ipiv + jay + 1 )

!                       IF ( i /= j .AND. PIVOT_set( j ) ) THEN
!                         write(6,*) ' diag ', j, ' pivot set '
!                       END IF

                        w1 = DIAG( j ) - A( jj ) * A( jj ) / A( apos )
!                       IF ( w1 <= 0.0 ) THEN
!                       IF ( W1 <= 1.0D-15 ) THEN
                        IF ( w1 <= addon ) THEN
                           IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) >= 2 )         &
                              WRITE( ICNTL( 2 ), 2020 ) j, w1
                           iphase = 2
                        END IF
                     END IF
                     onenrm = onenrm + ABS( A( jj ) )
                     rmax = MAX( ABS( A( jj ) ),rmax )
                     IF ( jay <= nass - ipiv ) THEN
                        IF ( ABS( A( jj ) ) > amax ) amax = ABS( A( jj ) )
                     END IF
                  END DO

!  Now calculate largest entry in other part of row.

                  apos1 = apos
                  kk = nfront - ipiv
                  IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) >= 2 )                  &
                    WRITE( ICNTL( 2 ), 2070 ) npiv, i, onenrm

! Jump if still in phase 1.

                  IF ( iphase /= 1 ) THEN

!  Check to see if pivot must be increased

                     IF ( A( apos ) < addon + onenrm ) THEN

!  Adjust diagonal entry and record change in DIAG

                        DIAG( i ) = addon + onenrm - A( apos )
!                       OMEGA( i ) = addon + onenrm - A( apos )
                        IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) >= 2 )            &
                          WRITE( ICNTL( 3 ), 2080 ) i, DIAG( i )
!                         WRITE( ICNTL( 3 ), 2080 ) i, OMEGA( i )
                        A( apos ) = onenrm + addon
                     ELSE
!                       OMEGA( i ) = zero
                        DIAG( i ) = zero
                     END IF
                  ELSE
!                    OMEGA( i ) = zero
                     DIAG( i ) = zero
                  END IF
!                 PIVOT_set( i ) = .TRUE.
                  pivsiz = 1 ; irow = ipiv - npiv

!  Pivot has been chosen. If block pivot of order 2, pivsiz is equal to 2,
!  otherwise pivsiz is 1. The following loop moves the pivot block to the top
!  left hand corner of the frontal matrix.

                  DO krow = 1, pivsiz
                     IF ( irow == 1 ) CYCLE
                     j1 = posfac + irow
                     j2 = posfac + nfront - npiv - 1
                     IF ( j2 >= j1 ) THEN
                        apos2 = apos + 1

!  Swap portion of rows whose column indices are greater than later row.

                        DO jj = j1, j2
                           swap = A( apos2 )
                           A( apos2 ) = A( jj )
                           A( jj ) = swap
                           apos2 = apos2 + 1
                        END DO
                     END IF
                     j1 = posfac + 1 ; j2 = posfac + irow - 2
                     apos2 = apos
                     kk = nfront - irow - npiv

!  Swap portion of rows/columns whose indices lie between the two rows.

                     DO jj = j2, j1, - 1
                        kk = kk + 1
                        apos2 = apos2 - kk
                        swap = A( apos2 )
                        A( apos2 ) = A( jj )
                        A( jj ) = swap
                     END DO
                     IF ( npiv /= 0 ) THEN
                        apos1 = posfac
                        kk = kk + 1
                        apos2 = apos2 - kk

!  Swap portion of columns whose indices are less than earlier row.

                        DO jj = 1, npiv
                           kk = kk + 1
                           apos1 = apos1 - kk ; apos2 = apos2 - kk
                           swap = A( apos2 )
                           A( apos2 ) = A( apos1 )
                           A( apos1 ) = swap
                        END DO
                     END IF

!  Swap diagonals and integer indexing information

                     swap = A( apos )
                     A( apos ) = A( posfac )
                     A( posfac ) = swap
                     ipos = iwpos + npiv + 2
                     iexch = iwpos + irow + npiv + 1
                     iswap = IW( ipos )
                     IW( ipos ) = IW( iexch )
                     IW( iexch ) = iswap
                  END DO


!  Perform the elimination using entry (ipiv,ipiv) as pivot.
!  We store u and D(inverse).

!  560            CONTINUE
                  A( posfac ) = one / A( posfac )
                  IF ( A( posfac ) < zero ) neig = neig + 1
                  j1 = posfac + 1 ; j2 = posfac + nfront - npiv - 1
                  IF ( j2 >= j1 ) THEN
                     ibeg = j2 + 1
                     DO jj = j1, j2
                        amult = - A( jj ) * A( posfac )

!  Update diag array.

                        j = IW( iwpos + npiv + jj - j1 + 3 )
!                       IF ( i /= j .AND. PIVOT_set( j ) ) THEN
!                         write(6,*) ' diag2', j, ' pivot set '
!                       END IF
                        DIAG( j ) = DIAG( j ) + amult * A( jj )
                        IF ( ICNTL( 2 ) > 0 .AND. ICNTL( 3 ) >= 2 )            &
                           WRITE( ICNTL( 2 ), 2090 ) j, DIAG( j )
                        iend = ibeg + nfront - ( npiv + jj - j1 + 2 )

!DIR$ IVDEP
                        DO irow = ibeg, iend
                           jcol = jj + irow - ibeg
                           A( irow ) = A( irow ) + amult * A( jcol )
                        END DO
                        ibeg = iend + 1
                        A( jj ) = amult
                     END DO
                  END IF
                  npiv = npiv + 1
                  ntotpv = ntotpv + 1
                  jpiv = 1
                  posfac = posfac + nfront - npiv + 1
               END DO
            END DO
            IF ( npiv /= 0 ) nblk = nblk + 1
            ioldps = iwpos ; iwpos = iwpos + nfront + 2
            IF ( npiv /= 0 ) THEN
               IF ( npiv <= 1 ) THEN
                  IW( ioldps ) = - IW( ioldps )
                  DO k = 1, nfront
                     j1 = ioldps + k
                     IW( j1 ) = IW( j1 + 1 )
                  END DO
                  iwpos = iwpos - 1
               ELSE
                  IW( ioldps + 1 ) = npiv

!  Copy remainder of element to top of stack

               END IF
            END IF
            liell = nfront - npiv
            IF ( liell /= 0 .AND. iass /= nsteps ) THEN

               IF( iwpos + liell >= istk )                                     &
                    CALL MDCHL_compress( A, IW, istk, istk2, iinput, 2,        &
                                         ncmpbr, ncmpbi )
               istk = istk - liell - 1
               IW( istk ) = liell
               j1 = istk ; kk = iwpos - liell - 1

! DIR$ IVDEP
               DO k = 1, liell
                  j1 = j1 + 1
                  kk = kk + 1
                  IW( j1 ) = IW( kk )
               END DO

!  We copy in reverse direction to avoid overwrite problems.

               laell = ( ( liell + 1 ) * liell ) / 2
               kk = posfac + laell
               IF ( kk == astk ) THEN
                  astk = astk - laell
               ELSE

!  The move and zeroing of array A is performed with two loops so
!  that they may be vectorized

                  kmax = kk - 1

!DIR$ IVDEP
                  DO k = 1, laell
                     kk = kk - 1
                     astk = astk - 1
                     A( astk ) = A( kk )
                  END DO
                  kmax = MIN( kmax, astk - 1 )
                  A( kk : kmax ) = zero
               END IF
               azero = MIN( azero, astk - 1 )
            END IF
            IF ( npiv == 0 ) iwpos = ioldps
         END DO

!  End of loop on tree nodes.

         IW( 1 ) = nblk
         IF ( ntwo > 0 ) IW( 1 ) = - nblk
         nrlbdu = posfac - 1 ; nirbdu = iwpos - 1

         IF ( ntotpv /= n ) THEN
            INFO( 1 ) = 3
            INFO( 2 ) = ntotpv
         END IF
         INFO( 9 ) = nrlbdu
         INFO( 10 ) = nirbdu
         INFO( 12 ) = ncmpbr
         INFO( 13 ) = ncmpbi
         INFO( 14 ) = ntwo
         INFO( 15 ) = neig
     RETURN

 2000 FORMAT( ' Diag ', 2ES24.16, /, '      ', 2ES24.16 )
 2010 FORMAT( ' Pivot has value ', ES24.16 )
 2020 FORMAT( ' Phase 2, j, w1 ', I6, ES24.16 )
!2030 FORMAT( ' IWPOS = ', I6, ' IW( IWPOS ),...,IW( IWPOS+NFONT+1 ) =',      &
!               /, ( 12I6 ) )
!2040 FORMAT ( ' *** Warning message from subroutine MCFC *** iflag =', I2,   &
!              /, ' pivot', I6,' has different sign from the previous one' )
 2050 FORMAT( ' Negative pivot encountered at stage', I8 )
 2060 FORMAT( ' Pivot, pivot row', 2I6 )
 2070 FORMAT( ' npiv, i, onenrm ', 3I6 )
 2080 FORMAT( ' i, Perturbation ', I6, ES12.4 )
 2090 FORMAT( ' j, DIAG  ', I6, ES12.4 )
 2100 FORMAT( ' Phase = ', I1 )

!  End of subroutine MDCHL_mcfc

     END SUBROUTINE MDCHL_mcfc

!-*-*-*-  L A N C E L O T  -B-  MDCHL_compress S U B R O U T I N E -*-*-*-*

     SUBROUTINE MDCHL_compress( A, IW, j1, j2, itop, ireal, ncmpbr, ncmpbi )

!  Compress the data structures

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER j1, j2, itop, ireal, ncmpbr, ncmpbi
     INTEGER, DIMENSION( * ) :: IW
     REAL ( KIND = wp ), DIMENSION( * ) :: A

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: jj, ipos
     
     ipos = itop - 1
     IF ( j2 /= ipos ) THEN
       IF ( ireal /= 2 ) THEN
         ncmpbr = ncmpbr + 1
         DO jj = j2, j1, - 1
           A( ipos ) = A( jj )
           ipos = ipos - 1
         END DO
       ELSE
         ncmpbi = ncmpbi + 1
         DO jj = j2, j1, - 1
           IW( ipos ) = IW( jj )
           ipos = ipos - 1
         END DO
       END IF
       j2 = itop - 1 ; j1 = ipos + 1
     END IF
     RETURN

!  End of subroutine MDCHL_compress

     END SUBROUTINE MDCHL_compress

!-*-*-*-  L A N C E L O T  -B-  MDCHL_idiag  F U N C T I O N -*-*-*-*

     FUNCTION MDCHL_idiag( ix, iy )

!  Obtain the displacement from the start of the assembled matrix (of order IX)
!  of the diagonal entry in its row IY

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER :: MDCHL_idiag
     INTEGER, INTENT( in ) :: ix, iy
     
     MDCHL_idiag = ( ( iy - 1 ) * ( 2 * ix - iy + 2 ) ) / 2
     RETURN

!  End of function MDCHL_idiag

     END FUNCTION MDCHL_idiag

!-*-*-*-  L A N C E L O T  -B-  MDCHL_syprc S U B R O U T I N E -*-*-*-*

     SUBROUTINE MDCHL_syprc( la, liw, A, IW, neg1, neg2 )

!   The Gill-Murray-Ponceleon-Saunders code for modifying the negative
!   eigen-components obtained when factorizing a symmetric indefinite
!   matrix using the HSL package MA27 (see SOL 90-8, P.19-21)

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( in ) :: la, liw
     INTEGER, INTENT( OUT ) :: neg1, neg2
     INTEGER, INTENT( in ), DIMENSION( liw ) :: IW
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( la ) :: A

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: alen, apos, iblk, nblk, ipos, nrows, ncols, j, k
     REAL ( KIND = wp ):: alpha, beta, gamma, tau
     REAL ( KIND = wp ):: t, co, si, e1, e2, epsmch
     LOGICAL :: one_by_one_block
     
     epsmch = EPSILON( one )

!  NEG1 and NEG2 are the number of negative eigenvalues which arise from
!  negative 1x1 and 2x2 block pivots

     neg1 = 0 ; neg2 = 0
     nblk = ABS( IW( 1 ) )
     ipos = 2
     apos = 1

!  Loop over all the block pivots

     DO iblk = 1, nblk
       ncols = IW( ipos )
       IF ( ncols < 0 ) THEN
         nrows = 1
         ncols = - ncols
       ELSE
         ipos = ipos + 1
         nrows = IW( ipos )
       END IF

!  Process the diagonals in this block

       alen = ncols
       one_by_one_block = .TRUE.
       DO k = ipos + 1, ipos + nrows
         IF ( one_by_one_block ) THEN
           alpha = A( apos )
           j = IW( k )
           one_by_one_block = j > 0
           IF ( one_by_one_block ) THEN

!  Negative 1x1 block

             IF ( alpha < zero ) THEN
               neg1 = neg1 + 1
               A( apos ) = -alpha
             ELSE
               IF ( alpha > one / epsmch ) THEN
                 neg1 = neg1 + 1
                 A( apos ) = one / epsmch
               END IF
             END IF
           ELSE
             beta = A( apos + 1 )
             gamma = A( apos + alen )

!  2x2 block: ( ALPHA  BETA  ) = ( C  S ) ( E1    ) ( C  S )
!             ( BETA   GAMMA )   ( S -C ) (    E2 ) ( S -C )

             IF ( alpha * gamma < beta ** 2 ) THEN
               tau = ( gamma - alpha ) / ( two * beta )
               t = - one / ( ABS( tau ) + SQRT( one + tau ** 2 ) )
               IF ( tau < zero ) t = -t
               co = one / SQRT( one + t ** 2 )
               si = t * co
               e1 = alpha + beta * t
               e2 = gamma - beta * t

!  Change E1 and E2 to their absolute values and then multiply the three
!  2 * 2 matrices to get the modified ALPHA, BETA AND GAMMA

               IF ( e1 < zero ) THEN
                  neg2 = neg2 + 1
                  e1 = - e1
               END IF
               IF ( e2 < zero ) THEN
                  neg2 = neg2 + 1
                  e2 = - e2
               END IF
               A( apos ) = co ** 2 * e1 + si ** 2 * e2
               A( apos + 1 ) = co * si * ( e1 - e2 )
               A( apos + alen ) = si ** 2 * e1 + co ** 2 * e2
             END IF
           END IF
         ELSE
           one_by_one_block = .TRUE.
         END IF
         apos = apos + alen
         alen = alen - 1
       END DO
       ipos = ipos + ncols + 1
     END DO
     
     RETURN

!  End of subroutine MDCHL_syprc

     END SUBROUTINE MDCHL_syprc

!-*-*-*-*-*-  L A N C E L O T  -B-  MDCHL_gmps  S U B R O U T I N E -*-*-*-*-*-

     SUBROUTINE MDCHL_gmps( n, rank, FACTORS, neg1, neg2, PERM, D )

!   The Gill-Murray-Ponceleon-Saunders code for modifying the negative
!   eigen-components obtained when factorizing a symmetric indefinite
!   matrix (see SOL 90-8, P.19-21) using the GALAHAD package SILS 

     USE GALAHAD_SILS_double, ONLY: SILS_factors, SILS_enquire, SILS_alter_d

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, rank
     INTEGER, INTENT( OUT ) :: neg1, neg2
     INTEGER, INTENT( OUT ), DIMENSION( n ) :: PERM
     TYPE ( SILS_factors ), INTENT( INOUT ) :: FACTORS
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 2, n ) :: D

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

     INTEGER :: i
     REAL ( KIND = wp ) :: alpha, beta, gamma, tau
     REAL ( KIND = wp ) :: t, c , s, e1, e2, eigen, eigen_zero
     LOGICAL :: oneby1
     
     eigen_zero = EPSILON( one )
     CALL SILS_enquire( FACTORS, PIVOTS = PERM, D = D )
     D( 1, rank + 1 : n ) = zero

!  neg1 and neg2 are the number of negative eigenvalues which arise
!  from small or negative 1x1 and 2x2 block pivots

     neg1 = 0 ; neg2 = 0 

!  Loop over all the block pivots

     oneby1 = .TRUE.
     DO i = 1, n

!  Decide if the current block is a 1x1 or 2x2 pivot block

       IF ( oneby1 ) THEN
         IF ( i < n ) THEN
           oneby1 = PERM( i ) > 0
         ELSE
           oneby1 = .TRUE.
         END IF
        
         alpha = D( 1, i )

!  =========
!  1x1 block
!  =========

         IF ( oneby1 ) THEN

!  Record the eigenvalue

           IF ( alpha /= zero ) THEN
             eigen = one / alpha
           ELSE
             eigen = zero
           END IF

!  Negative 1x1 block
!  ------------------

           IF ( eigen < - eigen_zero ) THEN 
             neg1 = neg1 + 1 
             D( 1, i ) = - alpha 

!  Small 1x1 block
!  ---------------

           ELSE IF ( eigen < eigen_zero ) THEN 
             neg1 = neg1 + 1 
             D( 1, i ) = one / eigen_zero 
           END IF 

!  =========
!  2x2 block
!  =========

         ELSE
         
           beta = D( 2, i )
           gamma = D( 1, i + 1 )

!  2x2 block: (  alpha  beta   ) = (  c  s  ) (  e1     ) (  c  s  )
!             (  beta   gamma  )   (  s -c  ) (     e2  ) (  s -c  )

           IF ( alpha * gamma < beta ** 2 ) THEN 
             tau = ( gamma - alpha ) / ( two * beta ) 
             t = - one / ( ABS( tau ) + SQRT( one + tau ** 2 ) ) 
             IF ( tau < zero ) t = - t 
             c = one / SQRT( one + t ** 2 ) ; s = t * c 
             e1 = alpha + beta * t ; e2 = gamma - beta * t 

!  Record the first eigenvalue

             eigen = one / e1

!  Change e1 and e2 to their modified values and then multiply the
!  three 2 * 2 matrices to get the modified alpha, beta and gamma

!  Negative first eigenvalue
!  -------------------------

             IF ( eigen < - eigen_zero ) THEN 
               neg2 = neg2 + 1 
               e1 = - e1

!  Small first eigenvalue
!  ----------------------

             ELSE IF ( eigen < eigen_zero ) THEN 
               neg2 = neg2 + 1 
               e1 = one / eigen_zero 
             END IF 

!  Record the second eigenvalue

             eigen = one / e2

!  Negative second eigenvalue
!  --------------------------

             IF ( eigen < - eigen_zero ) THEN 
               neg2 = neg2 + 1 
               e2 = - e2

!  Small second eigenvalue
!  -----------------------

             ELSE IF ( eigen < eigen_zero ) THEN 
               neg2 = neg2 + 1 
               e2 = one / eigen_zero 
             END IF 

!  Record the modified block

             D( 1, i ) = c ** 2 * e1 + s ** 2 * e2 
             D( 2, i ) = c * s * ( e1 - e2 ) 
             D( 1, i + 1 ) = s ** 2 * e1 + c ** 2 * e2 
           END IF 
         END IF
       ELSE
         oneby1 = .TRUE.
       END IF
     END DO

!  Register the (possibly modified) diagonal blocks

     CALL SILS_alter_d( FACTORS, D, i )

     RETURN  

!  End of subroutine MDCHL_gmps

     END SUBROUTINE MDCHL_gmps

!-*-*-*-*-  L A N C E L O T  -B-  MDCHL_block_type   F U N C T I O N  -*-*-*-*-

     FUNCTION MDCHL_block_type( n, rank, FACTORS, PERM, D )

!  Given the factorization  A = P L D L^T P^T,
!  of a symmetric matrix A, where 
!  D is a block diagonal matrix with 1x1 or 2x2 blocks,
!  set MDCHL_block_type to 1 if the matrix is positive definite,
!                          2 if the matrix is indefinite, and
!                          3 if the matrix is singular

     USE GALAHAD_SILS_double, ONLY: SILS_factors, SILS_enquire

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER :: MDCHL_block_type
     INTEGER, INTENT( IN ) :: n, rank
     INTEGER, INTENT( OUT ), DIMENSION( n ) :: PERM
     TYPE ( SILS_factors ), INTENT( IN ) :: FACTORS
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 2, n ) :: D

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

     INTEGER :: i
     REAL ( KIND = wp ) :: alpha, beta, gamma, tau
     REAL ( KIND = wp ) :: t, e1, e2, eigen, eigen_zero
     LOGICAL :: oneby1, singular, posdef

     eigen_zero = EPSILON( one )
     singular = rank /= n
     posdef = .NOT. singular
     CALL SILS_enquire( FACTORS, PIVOTS = PERM, D = D )
     D( 1, rank + 1 : n ) = zero

!  Loop over all the block pivots

     oneby1 = .TRUE.
     DO i = 1, n

!  Decide if the current block is a 1x1 or 2x2 pivot block

       IF ( oneby1 ) THEN
         IF ( i < n ) THEN
           oneby1 = PERM( i ) > 0
         ELSE
           oneby1 = .TRUE.
         END IF
         
         alpha = D( 1, i )

!  =========
!  1x1 block
!  =========

         IF ( oneby1 ) THEN

!  Record the eigenvalue

           IF ( alpha /= zero ) THEN
             eigen = one / alpha
           ELSE
             eigen = zero
           END IF

!  Negative 1x1 block
!  ------------------

           IF ( eigen < - eigen_zero ) THEN 
             posdef = .FALSE.

!  Small 1x1 block
!  ---------------

           ELSE IF ( eigen < eigen_zero ) THEN 
             singular = .TRUE.
           END IF 

!  =========
!  2x2 block
!  =========

         ELSE
         
           beta = D( 2, i )
           gamma = D( 1, i + 1 )

!  2x2 block: (  alpha  beta   ) = (  c  s  ) (  e1     ) (  c  s  )
!             (  beta   gamma  )   (  s -c  ) (     e2  ) (  s -c  )

           IF ( beta == zero ) THEN
             e1 = alpha ; e2 = gamma
           ELSE
             tau = ( gamma - alpha ) / ( two * beta ) 
             t = - one / ( ABS( tau ) + SQRT( one + tau ** 2 ) ) 
             IF ( tau < zero ) t = - t 
             e1 = alpha + beta * t ; e2 = gamma - beta * t 
           END IF

!  Record the first eigenvalue

           eigen = one / e1

!  Negative first eigenvalue
!  -------------------------

           IF ( eigen < - eigen_zero ) THEN 
             posdef = .FALSE.

!  Small first eigenvalue
!  ----------------------

           ELSE IF ( eigen < eigen_zero ) THEN 
             singular = .TRUE.
           END IF 

!  Record the second eigenvalue

           eigen = one / e2

!  Negative second eigenvalue
!  --------------------------

           IF ( eigen < - eigen_zero ) THEN 
             posdef = .FALSE.

!  Small second eigenvalue
!  -----------------------

           ELSE IF ( eigen < eigen_zero ) THEN 
             singular = .TRUE.
           END IF 
         END IF
       ELSE
         oneby1 = .TRUE.
       END IF
     END DO
     
     IF ( .NOT. posdef ) THEN
       MDCHL_block_type = 2
     ELSE
       IF ( singular ) THEN
         MDCHL_block_type = 3
       ELSE
         MDCHL_block_type = 1
       END IF
     END IF
     
     RETURN  

!  End of function MDCHL_block_type

     END FUNCTION MDCHL_block_type

!-*  L A N C E L O T  -B-  MDCHL_get_singular_direction  S U B R O U T I N E  *-

     SUBROUTINE MDCHL_get_singular_direction( n, rank, FACTORS, PERM, D,       &
                                              S, consis, CONTROL, info )

!  Given the factorization  A = P L D L^T P^T,
!  of an symmetric indefinite and singular matrix A, where 
!  D is a block diagonal matrix with 1x1 or 2x2 blocks,
!  compute a solution s to the system

!    A * s = rhs

!  if this system is consistent. Otherwise, compute a descent direction s for
!  the quadratic Q
!                  T
!  Q( x ) = 1/2 x^T A ^x - rhs^T x

!  i.e. a direction s such that A s = 0 and s^T rhs > 0
!  The RHS is input in S, and the solution returned in S

     USE GALAHAD_SILS_double, ONLY: SILS_factors, SILS_control, SILS_enquire,  &
                                    SILS_part_solve

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, rank
     INTEGER, INTENT( OUT ) :: info
     LOGICAL, INTENT( OUT ) :: consis
     INTEGER, INTENT( OUT ), DIMENSION( n ) :: PERM
     TYPE ( SILS_factors ), INTENT( INOUT ) :: FACTORS
     TYPE ( SILS_control ), INTENT( IN ) :: CONTROL
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: S
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 2, n ) :: D

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

     INTEGER :: i, i1, i2, dolid
     REAL ( KIND = wp ) :: alpha, beta, gamma, tau, s1, s2
     REAL ( KIND = wp ) :: t, co , si, e1, e2, eigen, eigen_zero, rhs_zero
     LOGICAL :: oneby1

     eigen_zero = EPSILON( one )
     rhs_zero = eigen_zero ** 0.75

!  Compute the matrix D

     CALL SILS_enquire( FACTORS, PIVOTS = PERM, D = D )
     D( 1, rank + 1 : n ) = zero

!  Compute w as the solution of the lower-triangular system

!      ( P L P^T ) * w = rhs

     CALL SILS_part_solve( FACTORS, CONTROL, 'L', S, info )

!  Now, consider the system

!      ( P D P^T ) v  = w
!
!  Since D is singular, either (a) the system is consistent, in which case
!  a suitable v may be found, or (b) it is inconsistent, in which case
!  we instead compute a vector v in the null-space of ( P D P^T )

!  Loop over all the block pivots

     dolid = 0
     oneby1 = .TRUE.
     DO i = 1, n

!  Decide if the current block is a 1x1 or 2x2 pivot block

       IF ( oneby1 ) THEN
         IF ( i < n ) THEN
           oneby1 = PERM( i ) > 0
         ELSE
           oneby1 = .TRUE.
         END IF
         
         alpha = D( 1, i )

!  =========
!  1x1 block
!  =========

         IF ( oneby1 ) THEN
           i1 = PERM( i )
           s1 = S( i1 )

!  Record the eigenvalue

           IF ( alpha /= zero ) THEN
             eigen = one / alpha
           ELSE
             eigen = zero
           END IF

!  Small 1x1 block
!  ---------------

           IF ( eigen >= - eigen_zero .AND. eigen < eigen_zero ) THEN 
             IF ( ABS( s1 ) > rhs_zero ) THEN
               dolid = 1 ;  EXIT
             ELSE
               S( i1 ) = zero
             END IF

!  Not so small 1x1 block
!  ----------------------

           ELSE
             S( i1 ) = alpha * s1
           END IF 

!  =========
!  2x2 block
!  =========

         ELSE
           i1 = - PERM( i ) ; i2 = PERM( i + 1 )
           s1 = S( i1 ) ;  s2 = S( i2 )
           beta = D( 2, i )
           gamma = D( 1, i + 1 )

!  2x2 block: (  alpha  beta   ) = (  c  s  ) (  e1     ) (  c  s  )
!             (  beta   gamma  )   (  s -c  ) (     e2  ) (  s -c  )

           IF ( beta == zero ) THEN
             e1 = alpha ; e2 = gamma
             co = one ; si = zero
           ELSE
             tau = ( gamma - alpha ) / ( two * beta ) 
             t = - one / ( ABS( tau ) + SQRT( one + tau ** 2 ) ) 
             IF ( tau < zero ) t = - t 
             co = one / SQRT( one + t ** 2 ) ; si = t * co
             e1 = alpha + beta * t ; e2 = gamma - beta * t 
           END IF

!  Small first eigenvalue
!  ----------------------

           eigen = one / e1
           IF ( eigen >= - eigen_zero .AND. eigen < eigen_zero ) THEN 
             IF ( ABS( co * s1 + si * s2 ) > rhs_zero ) THEN
               dolid = - 1 ;  EXIT
             END IF
           END IF 


!  Small second eigenvalue
!  -----------------------

           eigen = one / e2
           IF ( eigen >= - eigen_zero .AND. eigen < eigen_zero ) THEN 
             IF ( ABS( si * s1 - co * s2 ) > rhs_zero ) THEN
               dolid = - 2 ;  EXIT
             END IF
           END IF 

!  Not so small 2x2 block
!  ----------------------

           S( i1 ) = alpha * s1 + beta * s2
           S( i2 ) = beta * s1 + gamma * s2
         END IF
       ELSE
         oneby1 = .TRUE.
       END IF
     END DO

!  The system is inconsistent, Compute a suitable "solution"
!  Remember to ensure that s^T rhs > 0

     consis = dolid == 0
     IF ( .NOT. consis ) THEN
       S = zero
       IF ( dolid == 1 ) THEN
         IF ( s1 > zero ) THEN
           S( i1 ) = one
         ELSE
           S( i1 ) = - one
         END IF
       ELSE IF ( dolid == - 1 ) THEN
         IF ( s1 * co + s2 * si > zero ) THEN
           S( i1 ) = co
           S( i2 ) = si
         ELSE
           S( i1 ) = - co
           S( i2 ) = - si
         END IF
       ELSE IF ( dolid == - 2 ) THEN
         IF ( s1 * si - s2 * co > zero ) THEN
           S( i1 ) = si
           S( i2 ) = - co
         ELSE
           S( i1 ) = - si
           S( i2 ) = co
         END IF
       END IF
     END IF

!  Compute s as the solution of the upper-triangular system

!      ( P L^T P^T ) s = v

     CALL SILS_part_solve( FACTORS, CONTROL, 'U', S, info )

     END SUBROUTINE MDCHL_get_singular_direction

!-*-*-*-*-  L A N C E L O T  -B-  MDCHL_get_donc  S U B R O U T I N E   -*-*-*-

     SUBROUTINE MDCHL_get_donc( n, rank, FACTORS, PERM, D, number, next, S,    &
                                sas, CONTROL, info )

!  Given the factorization  A = P L D L^T P^T,
!  of an symmetric indefinite matrix A, where 
!  D is a block diagonal matrix with 1x1 or 2x2 blocks,
!  compute a direction of negative curvature, s, for A

!  Method:
!  -------
!  1 ) choose a negative eigenvalue of D, lambda, and its corresponding
!      eigenvector v
!  2 ) s is chosen as the solution of the upper triangular system

!      ( P L^T P^T ) s = P v

!  It then follows that

!       s^T A s = s^T P L D L^T P^T s  = v^T D v = lambda v^T v  < 0

     USE GALAHAD_SILS_double, ONLY: SILS_factors, SILS_control, SILS_enquire,  &
                                    SILS_part_solve

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, rank
     INTEGER, INTENT( INOUT ) :: number
     INTEGER, INTENT( OUT ) :: info
     REAL ( KIND = wp ), INTENT( out ) :: sas
     LOGICAL, INTENT( INOUT ) :: next
     INTEGER, INTENT( OUT ), DIMENSION( n ) :: PERM
     TYPE ( SILS_factors ), INTENT( INOUT ) :: FACTORS
     TYPE ( SILS_control ), INTENT( IN ) :: CONTROL
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: S
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 2, n ) :: D

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

     INTEGER :: i, i1, i2, inum, new_number
     REAL ( KIND = wp ) :: alpha, beta, gamma, tau, eigen_min, s1, s2
     REAL ( KIND = wp ) :: t, co , si, e1, e2, eigen, eigen_zero
     LOGICAL :: oneby1, e1_min, first, twoby2
     
     eigen_zero = EPSILON( one )
     CALL SILS_enquire( FACTORS, PIVOTS = PERM, D = D )
     D( 1, rank + 1 : n ) = zero

!  inum counts the encountered negative eigenvalues

     inum = 0

!  Loop over all the block pivots

     oneby1 = .TRUE.

!  Look for the smallest eigenvalue

     IF ( .NOT. next ) THEN
       eigen_min = zero
       DO i = 1, n

!  Decide if the current block is a 1x1 or 2x2 pivot block

         IF ( oneby1 ) THEN
           IF ( i < n ) THEN
             oneby1 = PERM( i ) > 0
           ELSE
             oneby1 = .TRUE.
           END IF
           
           alpha = D( 1, i )

!  =========
!  1x1 block
!  =========

           IF ( oneby1 ) THEN

!  Record the eigenvalue

             IF ( alpha /= zero ) THEN
               eigen = one / alpha
             ELSE
               eigen = zero
             END IF

!  Sufficiently negative 1x1 block
!  -------------------------------

             IF ( eigen < - eigen_zero ) inum = inum + 1
             IF ( eigen < eigen_min ) THEN
               eigen_min = eigen
               i1 = PERM( i )
               twoby2 = .FALSE.
               new_number = inum
             END IF

!  =========
!  2x2 block
!  =========

           ELSE
           
             beta = D( 2, i )
             gamma = D( 1, i + 1 )

!  2x2 block: (  alpha  beta   ) = (  co  si  ) (  e1     ) (  co  si  )
!             (  beta   gamma  )   (  si -co  ) (     e2  ) (  si -co  )

             IF ( alpha * gamma < beta ** 2 ) THEN 
               tau = ( gamma - alpha ) / ( two * beta ) 
               t = - one / ( ABS( tau ) + SQRT( one + tau ** 2 ) ) 
               IF ( tau < zero ) t = - t 
               co = one / SQRT( one + t ** 2 ) ; si = t * co 
               e1 = alpha + beta * t ; e2 = gamma - beta * t 

!  Record the smaller of the two eigenvalues

               e1_min = e1 <= e2
               IF ( e1_min ) THEN
                 eigen = one / e1
               ELSE
                 eigen = one / e2
               END IF

!  Negative smallest eigenvalue
!  ----------------------------

               IF ( eigen < - eigen_zero ) inum = inum + 1
               IF ( eigen < eigen_min ) THEN
                 eigen_min = eigen
                 i1 = - PERM( i )
                 i2 = PERM( i + 1 )
                 IF ( e1_min ) THEN
                   s1 = co ; s2 = si
                 ELSE
                   s1 = si ; s2 = - co
                 END IF
                 twoby2 = .TRUE.
                 new_number = inum
               END IF
             END IF 
           END IF
         ELSE
           oneby1 = .TRUE.
         END IF
       END DO
       number = new_number

! This time, look for the next eigenvalue in the list (or the first if there
! isn't a next!)

     ELSE
     
       first = .TRUE.
       DO i = 1, n

!  Decide if the current block is a 1x1 or 2x2 pivot block

         IF ( oneby1 ) THEN
           IF ( i < n ) THEN
             oneby1 = PERM( i ) > 0
           ELSE
             oneby1 = .TRUE.
           END IF
           
           alpha = D( 1, i )

!  =========
!  1x1 block
!  =========

           IF ( oneby1 ) THEN

!  Record the eigenvalue

             IF ( alpha /= zero ) THEN
               eigen = one / alpha
             ELSE
               eigen = zero
             END IF

!  Sufficiently negative 1x1 block
!  -------------------------------

             IF ( eigen < - eigen_zero ) THEN
               inum = inum + 1
      
               IF ( first .OR. inum == number + 1 ) THEN
                 twoby2 = .FALSE.
                 i1 = PERM( i )
                 eigen_min = eigen
                 IF ( first ) THEN
                   new_number = 1 ; first = .FALSE.
                 ELSE
                   new_number = number + 1
                 END IF
               END IF
             END IF

!  =========
!  2x2 block
!  =========

           ELSE
           
             beta = D( 2, i )
             gamma = D( 1, i + 1 )

!  2x2 block: (  alpha  beta   ) = (  co  si  ) (  e1     ) (  co  si  )
!             (  beta   gamma  )   (  si -co  ) (     e2  ) (  si -co  )

             IF ( alpha * gamma < beta ** 2 ) THEN 
               tau = ( gamma - alpha ) / ( two * beta ) 
               t = - one / ( ABS( tau ) + SQRT( one + tau ** 2 ) ) 
               IF ( tau < zero ) t = - t 
               co = one / SQRT( one + t ** 2 ) ; si = t * co 
               e1 = alpha + beta * t ; e2 = gamma - beta * t 

!  Record the smaller of the two eigenvalues

               e1_min = e1 <= e2
               IF ( e1_min ) THEN
                 eigen = one / e1
               ELSE
                 eigen = one / e2
               END IF

!  Negative smallest eigenvalue
!  ----------------------------

               IF ( eigen < - eigen_zero ) THEN
                 inum = inum + 1
                 IF ( first .OR. inum == number + 1 ) THEN
                   twoby2 = .TRUE.
                   i1 = - PERM( i )
                   i2 = PERM( i + 1 )
                   IF ( e1_min ) THEN
                     s1 = co ; s2 = si
                   ELSE
                     s1 = si ; s2 = - co
                   END IF
                   IF ( first ) THEN
                     new_number = 1 ; first = .FALSE.
                   ELSE
                     new_number = number + 1
                   END IF
                 END IF
               END IF
      
             END IF 
           END IF
         ELSE
           oneby1 = .TRUE.
         END IF
       END DO
       number = new_number

     END IF

!  Compute v, the eigenvector corresponding to the smallest
!  eigenvalue of D, and store P v in S

     S = zero
     
     IF ( twoby2 ) THEN
       S( i1 ) = s1
       S( i2 ) = s2
     ELSE
       S( i1 ) = one
     END IF
     sas = eigen_min

!  Compute S as the solution of the upper-triangular system

!      ( P L^T P^T ) s = P v

     CALL SILS_part_solve( FACTORS, CONTROL, 'U', S, info )

     RETURN  

!  End of subroutine MDCHL_get_donc

     END SUBROUTINE MDCHL_get_donc

!-*-*-*-  L A N C E L O T  -B-  MDCHL_iccga S U B R O U T I N E -*-*-*-*

     SUBROUTINE MDCHL_iccga( n , nz, A, INI, INJ   , iai   , iaj   ,           &
                             IK, IW, W , c , ICNTL, CNTL, INFO, KEEP )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, nz, iai, iaj
     REAL ( KIND = wp ), INTENT( INOUT ) :: c
     INTEGER, INTENT( INOUT ), DIMENSION( iai ) :: INI
     INTEGER, INTENT( INOUT ), DIMENSION( iaj ) :: INJ
     INTEGER, INTENT( OUT ), DIMENSION( n, 4 ) :: IK, IW
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( iaj ) :: A
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n, 3 ) :: W
     INTEGER, INTENT( INOUT ), DIMENSION( 12 ) :: KEEP
     INTEGER, INTENT( IN ), DIMENSION( 5 ) :: ICNTL
     INTEGER, INTENT( OUT ), DIMENSION( 10 ) :: INFO
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( 3 ) :: CNTL

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i , ii, ir, ki, kj, kk, kl    , kll, j, kpp, k,                &
                kp, kr, nz0   , nzp1  , ic1   , ic2   , idummy, ir1, ir2,      &
                ipd, lcol, lrow, ncp, nd, nual, nucl, nurl, iflag, lp, mp
     REAL ( KIND = wp ) :: addon , epsmch, diamax, a1, a2
                          
     LOGICAL :: phase2

!  Restore kept data

         nurl = KEEP( 1 )
         nucl = KEEP( 2 )
         nual = KEEP( 3 )
         lrow = KEEP( 4 )
         lcol = KEEP( 5 )
         ncp  = KEEP( 6 )
         nd   = KEEP( 7 )
         ipd  = KEEP( 8 )
         lp = ICNTL( 1 )
         mp = ICNTL( 2 )

! Check restrictions on input parameters

         IF ( n < 1 ) THEN
            IF ( lp > 0 ) WRITE( lp, 2030 )
            iflag = - 1
            GO TO 150
         END IF
         IF ( nz < n ) THEN
            IF ( lp > 0 ) WRITE( lp, 2040 )
            iflag = - 2
!        GO TO 150
         END IF
         IF ( iai < nz ) THEN
            IF ( lp > 0 ) WRITE( lp, 2050 )
            iflag = - 3
            GO TO 150
         END IF
         IF ( iaj < 2 * nz ) THEN
            IF ( lp > 0 ) WRITE( lp, 2060 )
            iflag = - 4
            GO TO 150
         END IF

! Initialize work arrays

         W = zero
         IK( : , : 3 ) = 0
         nual = 0
         iflag = 0
         nd = 0
         ncp = 0
         epsmch = EPSILON( one )
         diamax = epsmch

! Count number of elements

         DO k = 1, nz
            i = INI( k )
            j = INJ( k )
            IF ( i < 1 .OR. i > n .OR. j < i .OR. j > n ) THEN
               IF ( lp > 0 ) WRITE( lp, 2070 ) k, i, j
               iflag = - 5
               GO TO 150
            END IF

!  Check for double entries on the diagonal and move diagonal from A to W

            IF ( i == j ) THEN
               nd = nd + 1
               IF ( W( i, 1 ) /= zero ) THEN
                  IF ( mp > 0 ) WRITE( mp, 2010 ) i
                  iflag = 1
               END IF
               W( i, 1 ) = W( i, 1 ) + A( k )
               diamax = MAX( diamax,ABS( W( i, 1 ) ) )
            ELSE

!  Remove zeros

               IF ( A( k ) == zero ) THEN
                  nd = nd + 1
               ELSE
                  IK( i, 1 ) = IK( i, 1 ) + 1
                  IK( j, 2 ) = IK( j, 2 ) + 1
                  nual = nual + 1
                  A( nual ) = A( k )
                  INI( nual ) = i
                  INJ( nual ) = j
               END IF
            END IF
         END DO

!  nz0 is the number of off diagonal non-zeros

         nz0 = nz - nd
         lcol = nz0
         lrow = nz0
         phase2 = .FALSE.
         addon = diamax * epsmch ** 0.25

! Treat diagonal matrices specially

         IF ( nz0 == 0 ) THEN
            DO i = 1, n
               IK( i, 1 ) = 0
               IK( i, 2 ) = i
               IK( i, 3 ) = 0

!  Modify the diagonals if necessary

               IF ( W( i, 1 ) <= addon ) THEN
                  iflag = 2
                  W( i, 1 ) = MAX( addon,-W( i, 1 ) )
                  IF ( mp > 0 ) WRITE( mp, 2000 ) i
               END IF
               W( i, 1 ) = one / SQRT( W( i, 1 ) )
               W( i, 2 ) = one
            END DO
            GO TO 200
         END IF

!  Non-diagonal matrix. initialize IW( i, 1 ) and IW( i, 2 ) to point just
!  beyond where the last component of row/column i will be stored

         kj = iai - nz0 + 1
         ki = 1
         DO i = 1, n
            ki = ki + IK( i, 1 )
            kj = kj + IK( i, 2 )
            IW( i, 1 ) = ki
            IW( i, 2 ) = kj

!  Modify the diagonals if necessary

            IF ( W( i, 1 ) <= addon ) THEN
               iflag = 2
               phase2 = .TRUE.
               W( i, 1 ) = MAX( addon, - W( i, 1 ) )
               IF ( mp > 0 ) WRITE( mp, 2000 ) i
            END IF
            W( i, 1 ) = one / SQRT( W( i, 1 ) )
         END DO

! Reorder by rows using in-place sort algorithm.

!  IW( i, 1 ) contains the number of non-zeroes in row i of the input matrix.
!  Initialize IW( i, 1 ) to point just beyond where the last element of the
!  row will be stored

!  Save current entry

         DO i = 1, nz0
            ir1 = INI( i )

!  If ir1 < 0 the element is in place already

            IF ( ir1 >= 0 ) THEN
               ic1 = INJ( i )
               a1 = A( i )

!  Determine correct position

               ki = IW( ir1, 1 ) - 1
               DO idummy = 1, nz0
                  IF ( i == ki ) EXIT

!  Save contents of that position

                  ir2 = INI( ki )
                  ic2 = INJ( ki )

!  Store current entry

                  INI( ki ) = -ir1
                  INJ( ki ) = ic1
                  IW( ir1, 1 ) = ki
                  ir1 = ir2
                  ic1 = ic2

!  Make corresponding changes for reals if required

                  a2 = A( ki )
                  A( ki ) = a1
                  a1 = a2
                  ki = IW( ir1, 1 ) - 1
               END DO

!  If current entry is in place it is stored here

               IF ( idummy /= 1 ) THEN
                  A( ki ) = a1
                  INJ( ki ) = ic1
                  INI( ki ) = -ir1
               END IF
               IW( ir1, 1 ) = i
            END IF
         END DO

! Check for double entries while using the constructed row file to set
! up the column file and compress the rowfile

         kk = 0
         DO ir = 1, n
            kpp = IW( ir, 1 )
            IW( ir, 1 ) = kk + 1
            kll = kpp + IK( ir, 1 ) - 1

!  Load row ir into W( *, 3 ).

            DO k = kpp, kll
               j = INJ( k )
               IF ( W( j, 3 ) /= zero ) THEN
                  iflag = 1
                  IF ( mp > 0 ) WRITE( mp, 2020 ) ir, j
               END IF
               W( j, 3 ) = W( j, 3 ) + A( k )
            END DO

!  Reload row ir into arrays A and INJ and adjust ini.

            DO k = kpp, kll
               j = INJ( k )
               IF ( W( j, 3 ) == zero ) THEN
                  nd = nd + 1
                  lrow = lrow - 1
                  lcol = lcol - 1
                  IK( ir, 1 ) = IK( ir, 1 ) - 1
                  IK( j, 2 ) = IK( j, 2 ) - 1
               ELSE
                  kk = kk + 1
                  A( kk ) = W( j, 3 ) * W( ir, 1 ) * W( j, 1 )
                  INJ( kk ) = j
                  W( j, 3 ) = zero
                  kr = IW( j, 2 ) - 1
                  IW( j, 2 ) = kr
                  INI( kr ) = ir
               END IF
            END DO
         END DO
         IF ( iflag == 1 ) THEN

!  Zero unused locations in ini

            nz0 = nz - nd
            DO i = 1, n - 1
               INI( IW( i, 2 ) + IK( i, 2 ) : IW( i + 1, 2 ) - 1 ) = 0
            END DO
         END IF

!  Store input matrix

         nual = iaj + 1
         DO ii = 1, n
            IK( ii, 4 ) = IK( ii, 1 )
            i = n - ii + 1
            W( i, 2 ) = one
            kp = IW( i, 1 )
            kl = kp + IK( i, 1 ) - 1
            DO kk = kp, kl
               k = kp + kl - kk
               nual = nual - 1
               A( nual ) = A( k )
               INJ( nual ) = INJ( k )
            END DO
            IW( i, 1 ) = nual - nz0
         END DO

!  Set different parameters

         nurl = 0
         nzp1 = nz0 + 1
         nucl = IW( 1, 2 )
         nual = nual - nz0

!  Activate incomplete factorization

         KEEP( 1 ) = nurl
         KEEP( 2 ) = nucl
         KEEP( 3 ) = nual
         KEEP( 4 ) = lrow
         KEEP( 5 ) = lcol
         KEEP( 6 ) = ncp
         KEEP( 8 ) = ipd
         CALL MDCHL_iccgc( n, nz0, W( 1, 2 ), A( nzp1 ), INI, INJ( nzp1 ),    &
                           iai, iaj - nz0, IK, IW, IW( 1, 3 ), W( 1, 3 ),     &
                           iflag, c, phase2, ICNTL, CNTL, KEEP )
         nurl = KEEP( 1 )
         nucl = KEEP( 2 )
         nual = KEEP( 3 )
         lrow = KEEP( 4 )
         lcol = KEEP( 5 )
         ncp  = KEEP( 6 )
         ipd  = KEEP( 8 )

!        WRITE( 6, 3030 )
!        WRITE( 6, 3000 ) 1, W( 1, 2 ), IK( 1, 1 ), IK( 1, 2 )
!        WRITE( 6, 3000 ) N, W( N, 2 ), IK( N, 1 ), IK( N, 2 )
!        WRITE( 6, 3040 ) IAJ, NZP1-1
!        WRITE( 6, 3050 ) 1, A( NZP1 - 1 + 1 ), INJ( NZP1 - 1 + 1 )

!  The factorization is terminated

         kp = 1
         DO i = 1, n
            kl = kp + IK( i,4 ) - 1
            IF ( kp <= kl ) THEN
               INI( kp:kl ) = i
            END IF
            kp = kl + 1
         END DO
         GO TO 200

!  Unsuccesful entry

 150 CONTINUE
     IF ( lp > 0 ) WRITE( lp, 2080 )

!  Save data, and set exit parameters

 200 CONTINUE
     KEEP( 1 ) = nurl
     KEEP( 2 ) = nucl
     KEEP( 3 ) = nual
     KEEP( 4 ) = lrow
     KEEP( 5 ) = lcol
     KEEP( 6 ) = ncp
     KEEP( 7 ) = nd
     KEEP( 8 ) = ipd
     KEEP( 9 ) = iflag
     
     INFO( 1 ) = KEEP( 9 )
     INFO( 2 ) = KEEP( 4 )
     INFO( 3 ) = KEEP( 5 )
     INFO( 4 ) = KEEP( 6 )
     INFO( 5 ) = KEEP( 7 )
     INFO( 6 ) = KEEP( 8 )
   
     RETURN

!  Non-executable statements

 2000 FORMAT( //,' Warning from ICCGA. Diagonal element',I5,                  &
                 ' has been modififed to be positive' )
 2010 FORMAT( //,'+','Warning: more than one diagonal entry in row',I5 )
 2020 FORMAT( //,'+','Warning: there is more than one entry in row',I5,       &
                   ' and column',I5 )
 2030 FORMAT( //, 34X,' n  is out of range.' )
 2040 FORMAT( //, 34X,' nz is out of range.' )
 2050 FORMAT( //, 34X,'iai is out of range.' )
 2060 FORMAT( //, 34X,'iaj is out of range.' )
 2070 FORMAT( //, 34X,'Element',I7,' is in row',I5,' and column',I5 )
 2080 FORMAT( '+','Error return from ICCGA because' )
!3000 FORMAT( I6, ES12.4, 2I6 )
!3030 FORMAT( ' *** In ICCGA. D and IK ' )
!3040 FORMAT( ' *** In ICCGA IAJ and NZ0 are ', 2I6, ' A and INJ ' )
!3050 FORMAT( I6, ES12.4, I6 )

!  End of subroutine MDCHL_iccga

     END SUBROUTINE MDCHL_iccga

     SUBROUTINE MDCHL_iccgb( n, A, INJ, iaj, W, IK, B, INFO, KEEP )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, iaj
     INTEGER, INTENT( IN ), DIMENSION( iaj ) :: INJ
     INTEGER, INTENT( IN ), DIMENSION( n, 2 ) :: IK
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( iaj ) :: A
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: B
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n, 2 ) :: W
     INTEGER, DIMENSION( 12 ), INTENT( IN ) :: KEEP
     INTEGER, DIMENSION( 10 ), INTENT( OUT ) :: INFO

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: ic, iip, ipi, ir, k, kl, kp, iflag, lrow
     REAL ( KIND = wp ) :: bic, bir

!  Restore kept data

     lrow  = KEEP( 4 )
     iflag = KEEP( 9 )

!  Scale the rhs

     B = B * W( : , 1 )

!  Precondition the rhs

     kp = 1

!  Perform the forward substitution

     DO iip = 1, n
       ic = IK( iip, 2 )
       kl = kp + IK( ic, 1 ) - 1
       bic = B( ic )
       DO k = kp, kl
         ir = INJ( k )
         B( ir ) = B( ir ) - A( k ) * bic
       END DO
       kp = kl + 1
     END DO
     kl = lrow

!  Perform the back substitution

     DO ipi = 1, n
       iip = n + 1 - ipi
       ir = IK( iip, 2 )
       kp = kl - IK( ir, 1 ) + 1
!      bir = - SUM( A( kp : kl ) * B( INJ( kp : kl ) ) )
       bir = zero
       DO k = kp, kl
         bir = bir - A( k ) * B( INJ( k ) )
       END DO
       B( ir ) = B( ir ) / W( ir, 2 ) + bir
       kl = kp - 1
     END DO

!  Rescale the rhs

     B = B * W( : , 1 )

     INFO( 1 ) = iflag
     RETURN

!  End of subroutine MDCHL_iccgb

     END SUBROUTINE MDCHL_iccgb

!-*-*-*-  L A N C E L O T  -B-  MDCHL_iccgc S U B R O U T I N E -*-*-*-*

     SUBROUTINE MDCHL_iccgc( n , nz, D , A , INI   , INJ   , iai  , iaj,       &
                             IK, IP, IW, W , iflag , c     , phase2,           &
                             ICNTL, CNTL, KEEP )

     USE LANCELOT_HSL_routines, ONLY: MA61_compress

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, iai, iaj
     INTEGER, INTENT( INOUT ) :: iflag
     INTEGER, INTENT( INOUT ) :: nz
     REAL ( KIND = wp ), INTENT( INOUT ) :: c
     LOGICAL, INTENT( INOUT ) :: phase2
     INTEGER, INTENT( INOUT ), DIMENSION( iai ) :: INI
     INTEGER, INTENT( INOUT ), DIMENSION( iaj ) :: INJ
     INTEGER, INTENT( INOUT ), DIMENSION( n, 3 ) :: IK
     INTEGER, INTENT( OUT ), DIMENSION( n, 2 ) :: IW
     INTEGER, INTENT( INOUT ), DIMENSION( n, 2 ) :: IP
     REAL ( KIND = wp ), DIMENSION( n ) :: D
     REAL ( KIND = wp ), DIMENSION( iaj ) :: A
     REAL ( KIND = wp ), DIMENSION( n ) :: W
     INTEGER, INTENT( INOUT ), DIMENSION( 12 ) :: KEEP
     INTEGER, INTENT( IN ), DIMENSION( 5 ) :: ICNTL
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( 3 ) :: CNTL

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, ii, iip, il, in, ip1, ipdp1, ir, j, j1, jj, jp, k, kc, kk
     INTEGER :: kl, kl1, klc, klj, kll, klr, kp, kp2, kpc, kpi, kpj, kpp
     INTEGER :: kpr, kr, krl, ks, l, lfuldd, lfull, mcl, nc, nfill, nm1
     INTEGER :: nr, nrjp, nz0, nzc, nzi, mp
     INTEGER :: ipd, lcol, lrow, ncp, nual, nucl, nurl
     REAL :: alfa, b1, b2, pfill, pivt, dd
     REAL ( KIND = wp ) :: aa, al, epsmch, addon, onenrm
     LOGICAL :: change

!-----------------------------------------------
!   E x t e r n a l   F u n c t i o n s
!-----------------------------------------------

! restore kept data
 
     nurl = KEEP( 1 )
     nucl = KEEP( 2 )
     nual = KEEP( 3 )
     lrow = KEEP( 4 )
     lcol = KEEP( 5 )
     ncp  = KEEP( 6 )
     ipd  = KEEP( 8 )
     mp = ICNTL( 2 )
     dd = REAL( CNTL( 1 ), KIND = KIND( dd ) )

! Initialize local variables

         epsmch = EPSILON( one )
         addon = epsmch ** 0.25
         change = c > zero
         nz0 = nz
         ipd = n
         alfa = 1.0 / 0.9
         b1 = - 0.03
         b2 = 0.03
         nfill = iaj - nz0 - n
         mcl = lcol
         c = c ** 2

! Initialize IK( *, 3 )

         IK( : n, 3 ) = 0

! Set up linked lists of rows/columns with equal number of non-zeros

         DO i = 1, n
            nzi = IK( i, 1 ) + IK( i, 2 ) + 1
            in = IK( nzi, 3 )
            IK( nzi, 3 ) = i
            IW( i, 2 ) = in
            IW( i, 1 ) = 0
            IF ( in /= 0 ) IW( in, 1 ) = i
         END DO

! Start the elimination loop

         DO iip = 1, n

! Search rows with nrjp nonzeros

            DO nrjp = 1, n
               jp = IK( nrjp, 3 )
               IF ( jp > 0 ) EXIT
            END DO

! Row jp is used as pivot

! Remove rows/columns involved in elimination from ordering vectors

            DO l = 1, 2
               kpp = IP( jp,l )
               kll = IK( jp,l ) + kpp - 1
               DO k = kpp, kll
                  IF ( l == 1 ) THEN
                     j = INJ( k )
                  ELSE
                     j = INI( k )
                  END IF
                  il = IW( j, 1 )
                  in = IW( j, 2 )
                  IW( j, 2 ) = -1
                  IF ( in >= 0 ) THEN
                     IF ( il /= 0 ) THEN
                        IW( il, 2 ) = in
                     ELSE
                        nz = IK( j, 1 ) + IK( j, 2 ) + 1
                        IK( nz, 3 ) = in
                     END IF
                     IF ( in > 0 ) IW( in, 1 ) = il
                  END IF
               END DO
            END DO

! Remove jp from ordering vectors

            il = IW( jp, 1 )
            in = IW( jp, 2 )
            IW( jp, 2 ) = - 10
            IF ( in >= 0 ) THEN
               nz = IK( jp, 1 ) + IK( jp, 2 ) + 1
               IK( nz, 3 ) = in
               IF ( in > 0 ) IW( in, 1 ) = il
            END IF

! Store pivot

            IW( jp, 1 ) = - iip

! Compress row file if necessary

            IF ( lrow + IK( jp, 1 ) + IK( jp, 2 ) > iaj - n ) c = cmax
            IF ( nurl + IK( jp, 1 ) + IK( jp, 2 ) >= nual )                   &
               CALL MA61_compress( A, INJ, iaj, n, IK( : , 1 ), IP( : , 1 ),  &
                                   .TRUE., ncp, nucl, nual )
            kp = IP( jp, 1 )
            IP( jp, 1 ) = nurl + 1

! Remove jp from columns contained in the pivot row

            DO k = kp, IK( jp, 1 ) + kp - 1
               j = INJ( k )
               kpc = IP( j, 2 )
               nz = IK( j, 2 ) - 1
               IK( j, 2 ) = nz
               klc = kpc + nz
               IF ( klc <= kpc ) THEN
                  INI( kpc ) = 0
               ELSE
                  DO kc = kpc, klc
                     IF ( jp == INI( kc ) ) EXIT
                  END DO
                  INI( kc ) = INI( klc )
                  INI( klc ) = 0
               END IF
               lcol = lcol - 1
               nurl = nurl + 1
               INJ( nurl ) = j
               A( nurl ) = A( k )
               INJ( k ) = 0
            END DO

! Transform column part of pivot row to the row file

            kp2 = IP( jp, 2 )
            DO k = kp2, IK( jp, 2 ) + kp2 - 1
               nurl = nurl + 1
               lcol = lcol - 1
               i = INI( k )
               kpr = IP( i, 1 )
               klr = kpr + IK( i, 1 ) - 1
               DO kr = kpr, klr
                  IF ( jp == INJ( kr ) ) EXIT
               END DO
               INJ( kr ) = INJ( klr )
               A( nurl ) = A( kr )
               A( kr ) = A( klr )
               INJ( klr ) = 0
               IK( i, 1 ) = IK( i, 1 ) - 1
               INJ( nurl ) = i
               INI( k ) = 0
            END DO
            nzc = IK( jp, 1 ) + IK( jp, 2 )
            IK( jp, 1 ) = nzc
            IK( jp, 2 ) = 0

!  Unpack pivot row and control diagonal value

            kp = IP( jp, 1 )
            kl = kp + nzc - 1
            onenrm = zero
            DO k = kp, kl
               aa = A( k )
               onenrm = onenrm + ABS( aa )
               j = INJ( k )
               W( j ) = aa
            END DO
            IF ( phase2 ) THEN
               IF ( D( jp ) <= addon + onenrm ) THEN
                  iflag = 2
                  IF ( mp > 0 ) WRITE( mp, 2000 ) jp
                  D( jp ) = addon + onenrm
               END IF
            END IF
            IF ( kp <= kl ) THEN

! Perform row operations

               DO nc = 1, nzc
                  kc = IP( jp, 1 ) + nc - 1
                  ir = INJ( kc )
                  al = A( kc )/D( jp )

! Compress row file if necessary

                  IF ( lrow + IK( ir, 1 ) + IK( jp, 1 ) > iaj - n )      &
                     c = cmax
                  IF ( nurl + IK( ir, 1 ) + IK( jp, 1 ) >= nual )             &
                       CALL MA61_compress( A, INJ, iaj, n, IK( : , 1 ),       &
                                           IP( : , 1 ), .TRUE., ncp, nucl, nual)
                  kr = IP( ir, 1 )
                  krl = kr + IK( ir, 1 ) - 1

!  Scan the other row and change sign in IW for each common column number

                  DO ks = kr, krl
                     j = INJ( ks )
                     IF ( IW( j, 2 ) == - 1 ) THEN
                        IW( j, 2 ) = 1
                        A( ks ) = A( ks ) - al * W( j )
                     END IF
                  END DO

!  Scan pivot row for fills

                  DO ks = kp, kl
                     j = INJ( ks )

!  Only entries in the upper triangular part are considered

                     IF ( j >= ir ) THEN
                        IF ( IW( j, 2 ) /= 1 ) THEN
                           aa = - al * W( j )
                           IF ( ir == j ) THEN
                              D( ir ) = D( ir ) + aa
                              IF ( D( ir ) <= addon ) phase2 = .TRUE.
                              GO TO 205
                           END IF
                           IF ( aa * aa <= c * ABS( D( ir ) * D( j ) ) ) THEN
                              D( j ) = D( j ) + aa
                              D( ir ) = D( ir ) + aa
                              IF ( D( ir ) <= addon ) phase2 = .TRUE.
                              GO TO 205
                           END IF
                           lrow = lrow + 1
                           IK( ir, 1 ) = IK( ir, 1 ) + 1

!  If possible place the new element next to the present entry

!  See if there is room at the end of the entry

                           IF ( kr <= krl ) THEN
                              IF ( krl /= iaj ) THEN
                                 IF ( INJ( krl + 1 ) == 0 ) THEN
                                    krl = krl + 1
                                    INJ( krl ) = j
                                    A( krl ) = aa
                                    GO TO 170
                                 END IF
                              END IF

!  See if there is room ahead of present entry

                              IF ( kr == nual ) THEN
                                 nual = nual - 1
                              ELSE
                                 IF ( INJ( kr - 1 ) /= 0 ) GO TO 150
                              END IF
                              kr = kr - 1
                              IP( ir, 1 ) = kr
                              INJ( kr ) = j
                              A( kr ) = aa
                              GO TO 170

!  New entry has to be created

  150                      CONTINUE
                              DO kk = kr, krl
                                 nual = nual - 1
                                 INJ( nual ) = INJ( kk )
                                 A( nual ) = A( kk )
                                 INJ( kk ) = 0
                              END DO
                           END IF

!  Add the new element

                           nual = nual - 1
                           INJ( nual ) = j
                           A( nual ) = aa
                           IP( ir, 1 ) = nual
                           kr = nual
                           krl = kr + IK( ir, 1 ) - 1

!  Create fill in column file

  170                   CONTINUE
                           nz = IK( j, 2 )
                           k = IP( j, 2 )
                           kl1 = k + nz - 1
                           lcol = lcol + 1

!  If possible place new element at the end of present entry

                           IF ( nz /= 0 ) THEN
                              IF ( kl1 /= iai ) THEN
                                 IF ( INI( kl1 + 1 ) == 0 ) THEN
                                    INI( kl1 + 1 ) = ir
                                    GO TO 200
                                 END IF
                              END IF

!  If possible place element ahead of present entry

                              IF ( k == nucl ) THEN
                                 IF ( nucl == 1 ) GO TO 180
                                 nucl = nucl - 1
                              ELSE
                                 IF ( INI( k - 1 ) /= 0 ) GO TO 180
                              END IF
                              k = k - 1
                              INI( k ) = ir
                              IP( j, 2 ) = k
                              GO TO 200
                           END IF

!  New entry has to be created

  180                   CONTINUE
                           IF ( nz + 1 >= nucl ) THEN

!  Compress column file if there is not room for new entry

                              IF ( lcol + nz + 2 >= iai ) c = cmax
                              CALL MA61_compress( A, INI, iai, n, IK( : , 2 ),&
                                                  IP( : , 2 ), .FALSE.,       &
                                                  ncp, nucl, nual )
                              k = IP( j, 2 )
                              kl1 = k + nz - 1
                           END IF

!  Transfer old entry into new

                           DO kk = k, kl1
                              nucl = nucl - 1
                              INI( nucl ) = INI( kk )
                              INI( kk ) = 0
                           END DO

!  Add the new element

                           nucl = nucl - 1
                           INI( nucl ) = ir
                           IP( j, 2 ) = nucl
  200                   CONTINUE
                           IK( j, 2 ) = nz + 1
                        END IF
                     END IF
  205             CONTINUE
                     IW( j, 2 ) = -1
                  END DO
               END DO

!  Update ordering arrays

               DO k = kp, kl
                  j = INJ( k )
                  W( j ) = zero
                  A( k ) = A( k ) / D( jp )
                  nz = IK( j, 1 ) + IK( j, 2 ) + 1
                  in = IK( nz, 3 )
                  IW( j, 2 ) = in
                  IW( j, 1 ) = 0
                  IK( nz, 3 ) = j
                  IF ( in /= 0 ) IW( in, 1 ) = j
               END DO
               mcl = MAX( mcl, lcol )
               pivt = FLOAT( iip ) / FLOAT( n )

!  Give warning if available space is used too early

               IF ( c == cmax ) THEN
                  IF ( ipd < iip ) GO TO 240
                  ipd = iip
                  IF ( pivt > 0.9 ) GO TO 240
                  iflag = 4
                  IF ( mp > 0 ) WRITE( mp, 2010 ) iip
                  GO TO 240
               ELSE

!  Change C if necessary

                  IF ( change ) THEN
                     pfill = FLOAT( lrow - nz0 ) / FLOAT( nfill )
                     IF ( pivt < 9.0E-1 ) THEN
                        IF ( pfill > alfa * pivt + b1 ) THEN
                           IF ( pfill < alfa * pivt + b2 ) GO TO 240
                           c = 2.25D+0 * c
                        END IF
                        alfa = ( 1.0 - pfill ) / ( 0.9 - pivt )
                        b1 = pfill - pivt * alfa - 0.03
                        b2 = b1 + 0.06
                     END IF

!  If the matrix is full, stop the sparse analyze

                  END IF
               END IF
            END IF
  240       CONTINUE
            nr = n - iip
            lfull = nr * ( nr - 1 ) / 2
            lfuldd = IFIX( dd * FLOAT( lfull ) )
            IF ( lcol >= lfuldd .AND. nurl + lfull < iaj ) EXIT
         END DO

!  Sparse elimination loop terminates. Factorize the remaining full matrix

         ipd = iip
         c = SQRT( c )
         lcol = mcl
         IF ( .NOT. change ) c = - c

!  The order of the full matrix is nr.
!  Loop through rows in the active matrix and store row numbers in ini

         kk = 0
         DO i = 1, nr
            jp = IK( i, 3 )
  270       CONTINUE
            IF ( jp > 0 ) THEN
               kk = kk + 1
               INI( kk ) = jp
               jp = IW( jp, 2 )
               GO TO 270
            END IF
            IF ( kk == nr ) EXIT
         END DO

!  Make a sort of the row numbers in ini

         DO i = 1, nr - 1
            j1 = i + 1
            DO j = j1, nr
               IF ( INI( j ) <= INI( i ) ) THEN
                  jj = INI( i )
                  INI( i ) = INI( j )
                  INI( j ) = jj
               END IF
            END DO
         END DO
         DO i = 1, nr
            ii = INI( i )
            IW( ii, 1 ) = ( - ipd ) - i
         END DO

!  Make an ordered list of the pivots

         DO i = 1, n
            ir = -IW( i, 1 )
            IK( ir, 2 ) = i
         END DO

!  Move full matrix to the front and order

         ipdp1 = ipd + 1
         nm1 = n - 1
         IF ( ipdp1 <= nm1 ) THEN
            DO iip = ipdp1, nm1
               jp = IK( iip, 2 )
               kp = IP( jp, 1 )
               kl = kp + IK( jp, 1 ) - 1

!   Move row jp to W

               DO k = kp, kl
                  j = INJ( k )
                  INJ( k ) = 0
                  W( j ) = A( k )
               END DO

!  Compress file if necessary

               IF ( nurl + n - iip >= nual )                                   &
                  CALL MA61_compress( A, INJ, iaj, n, IK( : , 1 ), IP( : , 1 ),&
                                      .TRUE., ncp, nucl, nual )
               IP( jp, 1 ) = nurl + 1
               IK( jp, 1 ) = n - iip

!  Move rows and column indices into pivotal order

               DO i = iip + 1, n
                  j = IK( i, 2 )
                  nurl = nurl + 1
                  A( nurl ) = W( j )
                  INJ( nurl ) = j
                  W( j ) = zero
               END DO
            END DO
            lrow = nurl

!  Factorize the full matrix

            DO iip = ipdp1, nm1
               jp = IK( iip, 2 )
               kpi = IP( jp, 1 )

!  If necessary, modify the diagonal to ensure that it is positive

!              onenrm = SUM( ABS( A( kpi : kpi + IK( jp, 1 ) - 1 ) ) )
               onenrm = zero
               DO j = kpi, kpi + IK( jp, 1 ) - 1
                  onenrm = onenrm + ABS( A( j ) )
               END DO
               IF ( phase2 ) THEN
                  IF ( D( jp ) <= addon + onenrm ) THEN
                     D( jp ) = addon + onenrm
                     iflag = 2
                     IF ( mp > 0 ) WRITE( mp, 2000 ) jp
                  END IF
               END IF

!  Loop through the other row

               ip1 = iip + 1
               IF ( ip1 /= n ) THEN
                  DO j = ip1, nm1
                     jj = IK( j, 2 )
                     kpj = IP( jj, 1 )
                     klj = kpj + IK( jj, 1 ) - 1
                     al = A( kpi ) / D( jp )
                     D( jj ) = D( jj ) - al * A( kpi )
                     IF ( D( jj ) <= addon ) phase2 = .TRUE.
                     kk = kpi + 1
                     DO k = 1, klj - kpj + 1
                        A( kpj + k - 1 ) = A( kpj + k - 1 )                    &
                                           - al * A( kk + k - 1 )
                     END DO

!  Store factor and proceed to next row

                     A( kpi ) = al
                     kpi = kpi + 1
                  END DO
               END IF

!  Modify last diagonal entry

               jj = IK( n, 2 )
               al = A( kpi ) / D( jp )
               D( jj ) = D( jj ) - al * A( kpi )
               IF ( D( jj ) <= addon ) phase2 = .TRUE.
               A( kpi ) = al
            END DO
         END IF

!  If necessary, modify the diagonal to ensure that it is positive

         jj = IK( n, 2 )
         IF ( D( jj ) <= addon ) THEN
            D( jj ) = addon
            iflag = 2
            phase2 = .TRUE.
            IF ( mp > 0 ) WRITE( mp, 2000 ) jj
         END IF

!  Save kept data

     KEEP( 1 ) = nurl
     KEEP( 2 ) = nucl
     KEEP( 3 ) = nual
     KEEP( 4 ) = lrow
     KEEP( 5 ) = lcol
     KEEP( 6 ) = ncp
     KEEP( 8 ) = ipd

     RETURN

!  Non-executable statements

 2000 FORMAT( //,' Warning modification of zero or negative',              &
                    ' diagonal entry has been performed in location', I7 )
 2010 FORMAT( //,' Warning available space used at pivot step', I7 )

!  End of subroutine MDCHL_iccgc

     END SUBROUTINE MDCHL_iccgc

!  End of module LANCELOT_MDCHL

   END MODULE LANCELOT_MDCHL_double


