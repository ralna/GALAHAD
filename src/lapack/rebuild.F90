! THIS VERSION: GALAHAD 4.3 - 2024-02-03 AT 11:25 GMT.

!  read a file containg a subset of the reference blas, lapack, etc
!  written in fortran 77, and output a multi-precision version capable
!  of control via C preprocessing commands capable of setting 
!  integer and real types, and changing procedure names

   PROGRAM BUILD
     IMPLICIT NONE
     INTEGER, PARAMETER :: input = 5      ! for original file
     INTEGER, PARAMETER :: out = 6        ! for preprocessed file 
     INTEGER, PARAMETER :: hout = 25      ! for generated preprocessor header
     INTEGER, PARAMETER :: scratch = 26   ! for intermediate workings
     INTEGER, PARAMETER :: max_line = 400 ! maximum line length
     INTEGER, PARAMETER :: max_chars = 75 ! maximum # characters in a line
     CHARACTER ( len = 80 ) :: new_line
     CHARACTER ( len = max_line ) :: in_line, line
     CHARACTER ( len = 8 ) :: date
     CHARACTER ( len = 10 ) :: time
     CHARACTER ( len = 8 ) :: sub( 1000 )
     CHARACTER ( len = 8 ) :: fun( 1000 )
     CHARACTER ( len = 4 ) :: suff64( 4 ) = (/ '64  ', '__64', '    ', '_64 ' /)
     INTEGER :: i, ib, j, k, l, l_end, l_new, l_next, line_max, lsame
     INTEGER :: nsub, nfun, lost, nz, nz2
     LOGICAL :: proc, proc_end, external_line

#ifdef BLAS
     CHARACTER ( len = 4 ) :: refs = 'blas'
     CHARACTER ( len = 4 ) :: urefs = 'BLAS'
     OPEN( UNIT = hout, FILE = 'galahad_blas.h' )
#else
     CHARACTER ( len = 6 ) :: refs = 'lapack'
     CHARACTER ( len = 6 ) :: urefs = 'LAPACK'
     OPEN( UNIT = hout, FILE = 'galahad_lapack.h' )
#endif

!  ===============================================
!  first pass - find subroutine and function names
!  ===============================================

     nsub = 0 ; nfun = 0

!  read the input file

     DO

!  read the next line, exiting the loop once the end of file is reached 
!  or a read error occurs

       new_line = REPEAT( ' ', 80 )
       READ( input, "( A80 )", end = 100, err = 100 ) new_line
       IF ( new_line( 1 : 1 ) == '*' ) CYCLE
       l_end = LEN_TRIM( new_line )

!  hunt for and record procedure key words

       DO l = 1, l_end

!  key word is subroutine

         IF ( l + 15 <= l_end ) THEN
           IF ( new_line( l : l + 15 ) == '      SUBROUTINE' ) THEN
             DO i = l + 16, l_end
               IF ( new_line( i + 1 : i + 1 ) == '(' ) EXIT
             END DO
             nsub = nsub + 1
             sub( nsub ) = '        '
             sub( nsub ) = new_line( l + 17 : i )
           END IF
         END IF

!  key word is recursive subroutine

         IF ( l + 19 <= l_end ) THEN
           IF ( new_line( l : l + 19 ) == 'RECURSIVE SUBROUTINE' ) THEN
             DO i = l + 20, l_end
               IF ( new_line( i + 1 : i + 1 ) == '(' ) EXIT
             END DO
             nsub = nsub + 1
             sub( nsub ) = '        '
             sub( nsub ) = new_line( l + 21 : i )
           END IF
         END IF

!  key word is function

         IF ( l + 9 <= l_end ) THEN
           IF ( new_line( l : l + 9 ) == 'L FUNCTION' .OR.                     &
                new_line( l : l + 9 ) == 'N FUNCTION' .OR.                     &
                new_line( l : l + 9 ) == 'R FUNCTION' ) THEN
             DO i = l + 10, l_end
               IF ( new_line( i + 1 : i + 1 ) == '(' ) EXIT
             END DO

!  record procedure name

!            IF ( new_line( l + 11 : i ) /= 'IEECHK' ) THEN
!            IF ( new_line( l + 11 : i ) /= 'XERBLA' .AND.                     &
!                 new_line( l + 11 : i ) /= 'LSAME' ) THEN
               nfun = nfun + 1
               fun( nfun ) = '        '
               fun( nfun ) = new_line( l + 11 : i )
!            END IF
           END IF
         END IF
       END DO
     END DO

!  create the header file

 100 CONTINUE

#ifndef BLAS
     WRITE( hout, 1990 )
1990 FORMAT( '#include "galahad_blas.h"' )
#endif
     DO i = 1, 4
       SELECT CASE ( i )
       CASE( 1 )
         WRITE( hout, "( '#ifdef INTEGER_64', /,                               &
        &         '#define GALAHAD_', A, '_interface GALAHAD_', A,             &
        &         '_interface_64', /,                                          &
        &         '#ifdef NO_UNDERSCORE_INTEGER_64')" ) urefs, urefs
       CASE( 2 )
         WRITE( hout, "( '#elif DOUBLE_UNDERSCORE_INTEGER_64' )" )
       CASE( 3 )
         WRITE( hout, "( '#elif NO_SYMBOL_INTEGER_64' )" )
         CYCLE
       CASE( 4 )
         WRITE( hout, "( '#else' )" )
       END SELECT

!  loop over function names

       DO l = 1, nfun
         IF ( LEN_TRIM( fun( l ) ) == 4 ) THEN
           WRITE( hout, "( '#define ', A, '   ', A, A )" )                     &
             TRIM( fun( l ) ), TRIM( fun( l ) ), TRIM( suff64( i ) )
         ELSE IF ( LEN_TRIM( fun( l ) ) == 5 ) THEN
           WRITE( hout, "( '#define ', A, '  ', A, A )" )                      &
             TRIM( fun( l ) ), TRIM( fun( l ) ), TRIM( suff64( i ) )
         ELSE
           WRITE( hout, "( '#define ', A, ' ', A, A )" )                       &
             TRIM( fun( l ) ), TRIM( fun( l ) ), TRIM( suff64( i ) )
         END IF
       END DO

!  loop over subroutine names

       DO l = 1, nsub
         IF ( LEN_TRIM( sub( l ) ) == 4 ) THEN
           WRITE( hout, "( '#define ', A, '     ', A, A )" )                   &
             TRIM( sub( l ) ), TRIM( sub( l ) ), TRIM( suff64( i ) )
         ELSE IF ( LEN_TRIM( sub( l ) ) == 5 ) THEN
           WRITE( hout, "( '#define ', A, '    ', A, A )" )                    &
             TRIM( sub( l ) ), TRIM( sub( l ) ), TRIM( suff64( i ) )
         ELSE IF ( LEN_TRIM( sub( l ) ) == 7 ) THEN
           WRITE( hout, "( '#define ', A, '  ', A, A )" )                      &
             TRIM( sub( l ) ), TRIM( sub( l ) ), TRIM( suff64( i ) )
         ELSE IF ( LEN_TRIM( sub( l ) ) == 8 ) THEN
           WRITE( hout, "( '#define ', A, ' ', A, A )" )                       &
             TRIM( sub( l ) ), TRIM( sub( l ) ), TRIM( suff64( i ) )
         ELSE
           WRITE( hout, "( '#define ', A, '   ', A, A )" )                     &
             TRIM( sub( l ) ), TRIM( sub( l ) ), TRIM( suff64( i ) )
         END IF
       END DO
     END DO
     WRITE( hout, "( '#endif', /, '#endif' )" )
     CLOSE( UNIT = hout )

!  =================================
!  second pass - preprocess the file
!  =================================

     REWIND( input )
     OPEN( UNIT = scratch, STATUS = 'scratch' )

     CALL DATE_AND_TIME( DATE = date, TIME = time )
     WRITE( 6, 2000 ) date( 1 : 4 ), date( 5: 6 ), date( 7 : 8 ),              &
        time( 1 : 2 ), time( 3 : 4 ), refs, refs

!  read the input file

     l_new = 0 ; proc = .FALSE. ; proc_end = .FALSE. ; external_line = .FALSE.
     in_line = REPEAT( ' ', max_line )
     DO

!  read the next line, exiting the loop once the end of file is reached 
!  or a read error occurs

       new_line = REPEAT( ' ', 80 ) 
       READ( input, "( A80 )", end = 200, err = 200 ) new_line
       IF ( new_line( 1 : 1 ) == '*' ) CYCLE
       l_end = LEN_TRIM( new_line )

!  amalgamate continuation lines

       IF ( new_line( l_end : l_end ) == '&' ) THEN
         new_line( l_end : l_end ) = ' '
         IF ( l_new == 0 ) THEN
           l_end = LEN_TRIM( new_line ) + 1
           in_line = REPEAT( ' ', max_line ) 
           in_line( 1 : l_end ) = new_line( 1 : l_end )
         ELSE
           new_line = ADJUSTL( new_line )
           l_end = LEN_TRIM( new_line ) + 1
           in_line( l_new + 1 : l_new + l_end ) = new_line( 1 : l_end )
         END IF
         l_new = l_new + l_end
         CYCLE
       ELSE
         IF ( l_new == 0 ) THEN
           in_line = REPEAT( ' ', max_line ) 
           in_line( 1 : l_end ) = new_line( 1 : l_end )
         ELSE
           new_line = ADJUSTL( new_line )
           l_end = LEN_TRIM( new_line )
           in_line( l_new + 1 : l_new + l_end ) = new_line( 1 : l_end )
         END IF
         l_end = l_new + l_end ; l_new = 0
       END IF
!      write(6,*) ' ... ', in_line( 1 : l_end )

       nz = - 1

!  transform the line to accommodate multi-precision symbols, and
!  to replace archaic fortran 77 functions

!  TODO: replace
!    '= dum(1)' '=INT(dum(1),KIND=ip_)'
!    '= isgn' '= REAL(isgn,KIND=KIND(sgn)_)'
!    '= iws' '= REAL(iws,KIND=KIND(work(1)))
!    '= lwkopt' '= REAL(lwkopt,KIND=KIND(work(1)))
!    '= maxwrk' '= REAL(maxwrk,KIND=KIND(work(1)))
!    '= ws'    '= REAL(ws,KIND=KIND(work(1)))
!    'lwkopt = work(1)' 'lwkopt = INT(work(1),KIND=ip_)'
!
!  and modify
!    IF (tol>=zero .AND. n*tol*(sminl/smax)<=MAX(eps,hndrth*tol)) THEN
!    thresh = MAX(ABS(tol)*smax, maxitr*(n*(n*unfl)))
!    thresh = MAX(tol*sminoa, maxitr*(n*(n*unfl)))
!    wsize = MAX(wsize, 2*mn+work(2*mn+1))
!    wsize = mn + work(mn+1)
!    z(2*n+5) = hundrd*nfail/REAL(iter)

       line = REPEAT( ' ', max_line )
       k = 1 ; l_next = 0 ; lsame = 0
       DO l = 1, l_end
         IF ( in_line( l : l ) /= ' ' .AND. nz < 0 ) nz = l
         IF ( l < l_next ) CYCLE
         IF ( in_line( l : l + 7 ) == 'EXTERNAL' ) external_line = .TRUE.
         IF ( in_line( l : l + 4 ) == 'LSAME' ) lsame = lsame + 1
         IF ( in_line( l : l + 12 ) == 'MAX(0, ILAENV' ) THEN
           line( k : k + 9 ) = 'MAX(0_ip_, '
           l_next = l + 6 ; k = k + 10
         ELSE IF ( in_line( l : l + 12 ) == 'MAX(1, ILAENV' ) THEN
           line( k : k + 9 ) = 'MAX(1_ip_, '
           l_next = l + 6 ; k = k + 10
         ELSE IF ( in_line( l : l + 12 ) == 'MAX(2, ILAENV' ) THEN
           line( k : k + 9 ) = 'MAX(2_ip_, '
           l_next = l + 6 ; k = k + 10

         ELSE IF ( in_line( l : l + 8 ) == 'IEEECK(0,' ) THEN
           line( k : k + 12 ) = 'IEEECK(0_ip_,'
           l_next = l + 9 ; k = k + 13
         ELSE IF ( in_line( l : l + 8 ) == 'IEEECK(1,' ) THEN
           line( k : k + 12 ) = 'IEEECK(1_ip_,'
           l_next = l + 9 ; k = k + 13

         ELSE IF ( in_line( l : l + 8 ) == 'ILAENV(1,' ) THEN
           line( k : k + 12 ) = 'ILAENV(1_ip_,'
           l_next = l + 9 ; k = k + 13
         ELSE IF ( in_line( l : l + 8 ) == 'ILAENV(2,' ) THEN
           line( k : k + 12 ) = 'ILAENV(2_ip_,'
           l_next = l + 9 ; k = k + 13
         ELSE IF ( in_line( l : l + 8 ) == 'ILAENV(3,' ) THEN
           line( k : k + 12 ) = 'ILAENV(3_ip_,'
           l_next = l + 9 ; k = k + 13
         ELSE IF ( in_line( l : l + 8 ) == 'ILAENV(4,' ) THEN
           line( k : k + 12 ) = 'ILAENV(4_ip_,'
           l_next = l + 9 ; k = k + 13
         ELSE IF ( in_line( l : l + 8 ) == 'ILAENV(5,' ) THEN
           line( k : k + 12 ) = 'ILAENV(5_ip_,'
           l_next = l + 9 ; k = k + 13
         ELSE IF ( in_line( l : l + 8 ) == 'ILAENV(6,' ) THEN
           line( k : k + 12 ) = 'ILAENV(6_ip_,'
           l_next = l + 9 ; k = k + 13
         ELSE IF ( in_line( l : l + 8 ) == 'ILAENV(7,' ) THEN
           line( k : k + 12 ) = 'ILAENV(7_ip_,'
           l_next = l + 9 ; k = k + 13
         ELSE IF ( in_line( l : l + 8 ) == 'ILAENV(8,' ) THEN
           line( k : k + 12 ) = 'ILAENV(8_ip_,'
           l_next = l + 9 ; k = k + 13
         ELSE IF ( in_line( l : l + 8 ) == 'ILAENV(9,' ) THEN
           line( k : k + 12 ) = 'ILAENV(9_ip_,'
           l_next = l + 9 ; k = k + 13
         ELSE IF ( in_line( l : l + 9 ) == 'ILAENV(10,' ) THEN
           line( k : k + 13 ) = 'ILAENV(10_ip_,'
           l_next = l + 10 ; k = k + 14
         ELSE IF ( in_line( l : l + 9 ) == 'ILAENV(11,' ) THEN
           line( k : k + 13 ) = 'ILAENV(11_ip_,'
           l_next = l + 10 ; k = k + 14
         ELSE IF ( in_line( l : l + 9 ) == 'ILAENV(12,' ) THEN
           line( k : k + 13 ) = 'ILAENV(12_ip_,'
           l_next = l + 10 ; k = k + 14
         ELSE IF ( in_line( l : l + 9 ) == 'ILAENV(13,' ) THEN
           line( k : k + 13 ) = 'ILAENV(13_ip_,'
           l_next = l + 10 ; k = k + 14
         ELSE IF ( in_line( l : l + 9 ) == 'ILAENV(14,' ) THEN
           line( k : k + 13 ) = 'ILAENV(14_ip_,'
           l_next = l + 10 ; k = k + 14
         ELSE IF ( in_line( l : l + 9 ) == 'ILAENV(15,' ) THEN
           line( k : k + 13 ) = 'ILAENV(15_ip_,'
           l_next = l + 10 ; k = k + 14
         ELSE IF ( in_line( l : l + 9 ) == 'ILAENV(16,' ) THEN
           line( k : k + 13 ) = 'ILAENV(16_ip_,'
           l_next = l + 10 ; k = k + 14

         ELSE IF ( in_line( l : l + 5 ) == 'ROT(1,' ) THEN
           line( k : k + 8 ) = 'ROT(1_ip_'
           l_next = l + 5 ; k = k + 9

         ELSE IF ( in_line( l : l + 7 ) == 'LARFG(1,' ) THEN
           line( k : k + 10 ) = 'LARFG(1_ip_'
           l_next = l + 7 ; k = k + 11

         ELSE IF ( in_line( l : l + 6 ) == 'INTEGER' ) THEN
           line( k : k + 11 ) = 'INTEGER(ip_)'
           l_next = l + 7 ; k = k + 12
         ELSE IF ( in_line( l : l + 15 ) == 'DOUBLE PRECISION' ) THEN
           line( k : k + 8 ) = 'REAL(r8_)'
           l_next = l + 16 ; k = k + 9
         ELSE IF ( in_line( l : l + 10 ) == 'COMPLEX *16' ) THEN
           line( k : k + 11 ) = 'COMPLEX(c8_)'
           l_next = l + 11 ; k = k + 12
         ELSE IF ( l + 5 <= l_end .AND. in_line( l : l + 5 ) == ' REAL ' ) THEN
           line( k : k + 10 ) = ' REAL(r4_) '
           l_next = l + 6 ; k = k + 11
         ELSE IF ( in_line( l : l + 2 ) == '0E0' .OR.                          &
                   in_line( l : l + 2 ) == '0e0' ) THEN
           line( k : k + 4 ) = '0_r4_'
           l_next = l + 3 ; k = k + 5
         ELSE IF ( in_line( l : l + 2 ) == '0D0' .OR.                          &
                   in_line( l : l + 2 ) == '0d0' ) THEN
           line( k : k + 4 ) = '0_r8_'
           l_next = l + 3 ; k = k + 5
         ELSE IF ( in_line( l : l + 2 ) == '1E0' .OR.                          &
                   in_line( l : l + 2 ) == '1e0' ) THEN
           line( k : k + 4 ) = '1_r4_'
           l_next = l + 3 ; k = k + 5
         ELSE IF ( in_line( l : l + 2 ) == '1D0' .OR.                          &
                   in_line( l : l + 2 ) == '1d0' ) THEN
           line( k : k + 4 ) = '1_r8_'
           l_next = l + 3 ; k = k + 5
         ELSE IF ( in_line( l : l + 2 ) == '5E0' .OR.                          &
                   in_line( l : l + 2 ) == '5e0' ) THEN
           line( k : k + 4 ) = '5_r4_'
           l_next = l + 3 ; k = k + 5
         ELSE IF ( in_line( l : l + 2 ) == '5D0' .OR.                          &
                   in_line( l : l + 2 ) == '5d0' ) THEN
           line( k : k + 4 ) = '5_r8_'
           l_next = l + 3 ; k = k + 5
         ELSE IF ( in_line( l : l + 2 ) == 'E+0' .OR.                       &
                   in_line( l : l + 2 ) == 'e+0' ) THEN
           line( k : k + 3 ) = '_r4_'
           l_next = l + 3 ; k = k + 4
         ELSE IF ( in_line( l : l + 2 ) == 'D+0' .OR.                       &
                   in_line( l : l + 2 ) == 'd+0' ) THEN
           line( k : k + 3 ) = '_r8_'
           l_next = l + 3 ; k = k + 4
         ELSE IF ( in_line( l : l + 2 ) == '.E0' .OR.                          &
                   in_line( l : l + 2 ) == '.e0' ) THEN
           line( k : k + 5 ) = '.0_r4_'
           l_next = l + 3 ; k = k + 6
         ELSE IF ( in_line( l : l + 2 ) == '.D0' .OR.                          &
                   in_line( l : l + 2 ) == '.d0' ) THEN
           line( k : k + 5 ) = '.0_r8_'
           l_next = l + 3 ; k = k + 6

         ELSE IF ( in_line( l : l + 5 ) == '), 0, ' ) THEN
           line( k : k + 7 ) = '), 0_ip_'
           l_next = l + 4 ; k = k + 8
         ELSE IF ( in_line( l : l + 5 ) == '), 1, ' ) THEN
           line( k : k + 7 ) = '), 1_ip_'
           l_next = l + 4 ; k = k + 8
         ELSE IF ( in_line( l : l + 6 ) == '), -1, ' ) THEN
           line( k : k + 8 ) = '), -1_ip_'
           l_next = l + 5 ; k = k + 9
         ELSE IF ( in_line( l : l + 4 ) == '), 0)' ) THEN
           line( k : k + 8 ) = '), 0_ip_)'
           l_next = l + 5 ; k = k + 9
         ELSE IF ( in_line( l : l + 4 ) == '), 1)' ) THEN
           line( k : k + 8 ) = '), 1_ip_)'
           l_next = l + 5 ; k = k + 9
         ELSE IF ( in_line( l : l + 5 ) == '), -1)' ) THEN
           line( k : k + 9 ) = '), -1_ip_)'
           l_next = l + 6 ; k = k + 10

         ELSE IF ( in_line( l : l + 4 ) == '( 0, ' ) THEN
           line( k : k + 6 ) = '( 0_ip_'
           l_next = l + 3 ; k = k + 7
         ELSE IF ( in_line( l : l + 4 ) == ', 0, ' ) THEN
           line( k : k + 6 ) = ', 0_ip_'
           l_next = l + 3 ; k = k + 7
         ELSE IF ( in_line( l : l + 3 ) == ', 0)' ) THEN
           line( k : k + 7 ) = ', 0_ip_)'
           l_next = l + 4 ; k = k + 8

         ELSE IF ( in_line( l : l + 4 ) == '( 1, ' ) THEN
           line( k : k + 6 ) = '( 1_ip_'
           l_next = l + 3 ; k = k + 7
         ELSE IF ( in_line( l : l + 4 ) == ', 1, ' ) THEN
           line( k : k + 6 ) = ', 1_ip_'
           l_next = l + 3 ; k = k + 7
         ELSE IF ( in_line( l : l + 3 ) == '  1,' ) THEN
           line( k : k + 7 ) = '  1_ip_,'
           l_next = l + 4 ; k = k + 8
         ELSE IF ( in_line( l : l + 3 ) == '  1)' ) THEN
           line( k : k + 7 ) = '  1_ip_)'
           l_next = l + 4 ; k = k + 8
         ELSE IF ( in_line( l : l + 5 ) == '( -1, ' ) THEN
           line( k : k + 7 ) = '( -1_ip_'
           l_next = l + 4 ; k = k + 8
         ELSE IF ( in_line( l : l + 5 ) == ', -1, ' ) THEN
           line( k : k + 7 ) = ', -1_ip_'
           l_next = l + 4 ; k = k + 8
         ELSE IF ( in_line( l : l + 4 ) == ', -1)' ) THEN
           line( k : k + 8 ) = ', -1_ip_)'
           l_next = l + 5 ; k = k + 9
         ELSE IF ( in_line( l : l + 4 ) == ', -1 ' ) THEN
           line( k : k + 8 ) = ', -1_ip_ '
           l_next = l + 5 ; k = k + 9

         ELSE IF ( in_line( l : l + 4 ) == '  -1,' ) THEN
           line( k : k + 7 ) = '  -1_ip_'
           l_next = l + 4 ; k = k + 8
         ELSE IF ( in_line( l : l + 4 ) == '  -1)' ) THEN
           line( k : k + 8 ) = '  -1_ip_)'
           l_next = l + 5 ; k = k + 9

         ELSE IF ( in_line( l : l + 3 ) == '(2, ' ) THEN
           line( k : k + 6 ) = '( 2_ip_'
           l_next = l + 2 ; k = k + 7
         ELSE IF ( in_line( l : l + 4 ) == '( 2, ' ) THEN
           line( k : k + 6 ) = '( 2_ip_'
           l_next = l + 3 ; k = k + 7
         ELSE IF ( in_line( l : l + 4 ) == ', 2, ' ) THEN
           line( k : k + 6 ) = ', 2_ip_'
           l_next = l + 3 ; k = k + 7
         ELSE IF ( in_line( l : l + 3 ) == ', 2)' ) THEN
           line( k : k + 7 ) = ', 2_ip_)'
           l_next = l + 4 ; k = k + 8

         ELSE IF ( in_line( l : l + 3 ) == '(3, ' ) THEN
           line( k : k + 6 ) = '( 3_ip_'
           l_next = l + 2 ; k = k + 7
         ELSE IF ( in_line( l : l + 4 ) == '( 3, ' ) THEN
           line( k : k + 6 ) = '( 3_ip_'
           l_next = l + 3 ; k = k + 7
         ELSE IF ( in_line( l : l + 4 ) == ', 3, ' ) THEN
           line( k : k + 6 ) = ', 3_ip_'
           l_next = l + 3 ; k = k + 7
         ELSE IF ( in_line( l : l + 3 ) == ', 3)' ) THEN
           line( k : k + 7 ) = ', 3_ip_)'
           l_next = l + 4 ; k = k + 8

         ELSE IF ( in_line( l : l + 3 ) == '(4, ' ) THEN
           line( k : k + 6 ) = '( 4_ip_'
           l_next = l + 2 ; k = k + 7
         ELSE IF ( in_line( l : l + 4 ) == '( 4, ' ) THEN
           line( k : k + 6 ) = '( 4_ip_'
           l_next = l + 3 ; k = k + 7
         ELSE IF ( in_line( l : l + 4 ) == ', 4, ' ) THEN
           line( k : k + 6 ) = ', 4_ip_'
           l_next = l + 3 ; k = k + 7
         ELSE IF ( in_line( l : l + 3 ) == ', 4)' ) THEN
           line( k : k + 7 ) = ', 4_ip_)'
           l_next = l + 4 ; k = k + 8

         ELSE IF ( in_line( l : l + 4 ) == '(16, ' ) THEN
           line( k : k + 7 ) = '( 16_ip_'
           l_next = l + 3 ; k = k + 8

         ELSE IF ( in_line( l : l + 6 ) == '1,2,3,4' ) THEN
           line( k : k + 22 ) = '1_ip_,2_ip_,3_ip_,4_ip_'
           l_next = l + 7 ; k = k + 23
         ELSE IF ( in_line( l : l + 7 ) == '-1,-1,-1' ) THEN
           line( k : k + 19 ) = '-1_ip_,-1_ip_,-1_ip_'
           l_next = l + 8 ; k = k + 20
         ELSE IF ( in_line( l : l + 4 ) == '-1,-1' ) THEN
           line( k : k + 12 ) = '-1_ip_,-1_ip_'
           l_next = l + 5 ; k = k + 13
         ELSE IF ( in_line( l : l + 3 ) == ',-1)' ) THEN
           line( k : k + 7 ) = ',-1_ip_)'
           l_next = l + 4 ; k = k + 8
         ELSE IF ( in_line( l : l + 3 ) == ', 1)' ) THEN
           line( k : k + 7 ) = ', 1_ip_)'
           l_next = l + 4 ; k = k + 8


         ELSE IF ( in_line( l : l + 3 ) == 'd,1)' ) THEN
           line( k : k + 8 ) = 'd, 1_ip_)'
           l_next = l + 4 ; k = k + 9

         ELSE
           line( k : k ) = in_line( l : l )
           l_next = l + 1 ; k = k + 1
         END IF
       END DO

!  hunt for key words

       DO l = 1, l_end

!  key word is subroutine

         IF ( l + 15 <= l_end ) THEN
           IF ( line( l : l + 15 ) == '      SUBROUTINE' )  THEN
             WRITE( out, "( '' )" ) ; proc = .TRUE.
           END IF
         END IF

!  key word is recursive subroutine

         IF ( l + 19 <= l_end ) THEN
           IF ( line( l : l + 19 ) == 'RECURSIVE SUBROUTINE' )  THEN
             WRITE( out, "( '' )" ) ; proc = .TRUE.
           END IF
         END IF

!  key word is function

         IF ( l + 9 <= l_end ) THEN
           IF ( line( l : l + 9 ) == ') FUNCTION' ) THEN
             WRITE( out, "( '' )" ) ; proc = .TRUE.
           END IF
         END IF

!  key word is function (with two spaces before)

         IF ( l + 9 <= l_end ) THEN
           IF ( line( l : l + 9 ) == '  FUNCTION' ) THEN
             WRITE( out, "( '' )" ) ; proc = .TRUE.
           END IF
         END IF

!  key word is logical function

         IF ( l + 15 <= l_end ) THEN
           IF ( line( l : l + 15 ) == 'LOGICAL FUNCTION' ) THEN
             WRITE( out, "( '' )" ) ; proc = .TRUE.
           END IF
         END IF

!  key word is call

         IF ( l + 5 <= l_end ) THEN
           IF ( line( l : l + 5 ) == ' CALL ' ) THEN
             lost = lost + 4
           END IF
         END IF
       END DO

!  record when a procedure line ends in a closing bracket

       IF ( proc ) THEN
         IF ( in_line( l_end : l_end ) == ')') THEN
           proc = .FALSE. ; proc_end = .TRUE.
         END IF
       END IF

!  remove all leading zeros

       nz2 = nz + 2
       line = ADJUSTL( line )
       l_end = LEN_TRIM( line )

!  output line if its length is no more than 76 characters (fewer if
!  corresponds to an external statement)

       IF ( external_line ) THEN
         line_max = max_chars - nz - 16
       ELSE IF ( lsame > 1 ) THEN
         line_max = max_chars - nz - 4 * ( lsame - 1 )
       ELSE
         line_max = max_chars - nz
       END IF

       DO 
         IF ( l_end <= line_max ) THEN
           IF ( .NOT. ( l_end == 1 .AND. line( 1 : 1 ) == '&' ) )              &
             WRITE( out, "( A, A )" ) REPEAT( ' ', nz ), TRIM( line )
           EXIT

!  the line has more than max_chars characters, split it into two 
!  appropriate-sized chunks, and repeat

         ELSE
           DO k = line_max, 1, - 1
!            IF ( line( k : k ) == ' ' .OR.                                    &
             IF ( ( line( k : k ) == ' ' .AND. line( k-1 : k-1 ) /= "'" ) .OR. &
                  line( k : k ) == ',' ) THEN
               IF ( line( k : k ) == ',' ) THEN
                 IF ( external_line .OR. lsame > 1 ) THEN
                   WRITE( out, "( A, A, '& ' )" )                              &
                     REPEAT( ' ', nz ), TRIM( line( 1 : k ) )
                 ELSE
                   WRITE( out, "( A, A, A, '&' )" )                            &
                     REPEAT( ' ', nz ), TRIM( line( 1 : k ) ),                 &
                     REPEAT( ' ', MAX( 0, max_chars + 1 - nz - k ) )
                 END IF
               ELSE
                 IF ( external_line .OR. lsame > 1 ) THEN
                   WRITE( out, "( A, A, ' &' )" )                              &
                     REPEAT( ' ', nz ), TRIM( line( 1 : k ) )
                 ELSE
                   WRITE( out, "( A, A, A, '&' )" )                            &
                     REPEAT( ' ', nz ), TRIM( line( 1 : k ) ),                 &
                     REPEAT( ' ', max_chars + 2 - nz - k )
                 END IF
               END IF
               nz = MIN( nz + 2, nz2 ) 
               EXIT
             END IF
           END DO
           line( 1 : k ) = REPEAT( ' ', k )
           line = ADJUSTL( line )
           l_end = LEN_TRIM( line )
         END IF
       END DO  
       external_line = .FALSE.

!  if a procedure line ends in a closing bracket, add a USE statement

       IF ( proc_end ) THEN
         WRITE( out, "( '          USE GALAHAD_KINDS' )" )
         proc_end = .FALSE.
       END IF
     END DO
 200 CONTINUE

     STOP
2000 FORMAT( '! THIS VERSION: GALAHAD 4.3 - ', A4, '-', A2, '-', A2,           &
       ' AT ', A2, ':', A2, ' GMT', //,  '#include "galahad_', A, '.h"', //,   &
       '! Reference ', A, ', see http://www.netlib.org/lapack/explore-html/' )
   END PROGRAM BUILD
