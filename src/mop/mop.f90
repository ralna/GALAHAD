! THIS VERSION: GALAHAD 2.6 - 23/11/2016 AT 071:55 GMT.

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*                             *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*         mop  M O D U L E    *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*                             *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Daniel Robinson

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_MOP_double


!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!                                                                             !
!  Feb. 1, 2008 : This module contains subroutines that perform a variety of  !
!                 matrix computations on matrices of type SMT_type.           !
!                                                                             !
!  Contains:                                                                  !
!              mop_Ax, mop_getval, mop_row_1_norms, mop_row_2_norms,          !
!              mop_row_infinity_norms, mop_column_2_norms, mop_scaleA          !
!                                                                             !
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   USE GALAHAD_SMT_double

   IMPLICIT NONE

   PRIVATE
   PUBLIC :: mop_Ax, mop_getval, mop_row_2_norms, mop_row_one_norms,           &
             mop_row_infinity_norms, mop_column_2_norms, mop_scaleA

!  Define the working precision to be double

   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

 CONTAINS


!-*-*-*-*-*  B E G I N  m o p _ A X   S U B R O U T I N E  *-*-*-*-*-*-*-*-*

   SUBROUTINE mop_Ax( alpha, A, X, beta, R, out, error,                        &
                      print_level, symmetric, transpose, m_matrix, n_matrix )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................................
!      .                                                            .
!      .  Perform the operation   R := alpha * A * x + beta * r     .
!      .                     or   R := alpha * A^T * x + beta * r   .
!      .                                                            .
!      ..............................................................

!  Arguments:
!  =========
!   alpha    a scalar variable of type real.
!
!   beta     a scalar variable of type real.
!            If beta is zero, R need not be initialized.
!
!   A        a scalar variable of derived type SMT_type.
!
!   R        the result r of either:
!
!               R :=  alpha * A * x + beta * r
!                            or
!               R :=  alpha * A^T * x + beta * r
!
!   X        the vector x.
!
!   symmetric (optional)
!            is a scalar variable of type logical.  If
!            symmetric = .TRUE., then the matrix A is assumed symmetric.
!            If symmetric = .FALSE. or is not present, then the matrix
!            A is assumed to NOT be symmetric.
!
!   transpose (optional)
!            is a scalar variable of type logical.
!            Possible values are:
!
!               transpose = .TRUE.      r <- alpha * A^T * x  + beta * x
!               transpose = .FALSE.     r <- alpha * A * x + beta * x
!
!            If transpose is not present, then "not-transposed" is assumed.
!
!   out     (optional) is a scalar variable of type integer, which
!           holds the stream number for informational messages;  the
!           file is assumed to already have been opened.  Default
!           is out = 6.
!
!   error   (optional) is a scalar variable of type integer, which
!           holds the stream number for error messages; the file
!           is assumed to already have been opened.  Default is error = 6.
!
!   print_level  (optional) is scalar variable of type integer.  It
!                controls the amount of printed information to unit
!                number defined by out.  Values and their meanings are:
!
!                print_level <= 0    Nothing.
!                print_level  = 1    The following is printed:
!                                    symmetric, transpose, A%m, A%n, A%type,
!                                    A%id, alpha, and beta.
!                print_level  = 2    Those from above, in additon to:
!                                    A%ptr, A%val, A%row, and A%col.
!                print_level >= 3    Those from above, in addition to:
!                                    Initial X, inital R, and final R.
!
!   m_matrix (optional) is a scalar variable of type integer, which overrides
!            the row dimension provided in A
!
!   n_matrix (optional) is a scalar variable of type integer, which overrides
!            the column dimension provided in A
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


!  Dummy arguments

     REAL ( KIND = wp ), INTENT( IN ) :: alpha, beta
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( : ) :: X
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: R
     TYPE( SMT_type), INTENT( IN  ) :: A
     INTEGER, INTENT( IN ), OPTIONAL :: error, out, print_level
     LOGICAL, INTENT( IN ), OPTIONAL :: transpose, symmetric
     INTEGER, INTENT( IN ), OPTIONAL :: m_matrix, n_matrix

! Set Parameters

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one  = 1.0_wp

!  Local variables

     INTEGER :: e_dev, i_dev, printing
     INTEGER :: i, j, nA
     INTEGER :: m, n, Acoli, Arowi
     REAL ( KIND = wp ) :: Xi, Xj, ri, rj, Avali
     LOGICAL :: trans, symm

! Determine values for optional parameters.

     IF ( PRESENT( error ) ) THEN
        e_dev = error
     ELSE
        e_dev = 6
     END IF

     IF ( PRESENT( out ) ) THEN
        i_dev = out
     ELSE
        i_dev = 6
     END IF

     IF ( PRESENT( print_level ) ) THEN
        printing = print_level
     ELSE
        printing = 0
     END IF

     IF ( PRESENT( transpose ) ) THEN
        trans = transpose
     ELSE
        trans = .FALSE.
     END IF

     IF ( PRESENT( symmetric ) ) THEN
        symm = symmetric
     ELSE
        symm = .FALSE.
     END IF

! Print Header

     IF ( printing >= 1 ) WRITE( i_dev, 1005 )

! Set some convenient variables.

     IF ( PRESENT( n_matrix ) ) THEN
       n = n_matrix
     ELSE
       n = A%n
     END IF
     IF ( PRESENT( m_matrix ) ) THEN
       m = m_matrix
     ELSE
       IF ( symm ) THEN
         m = n
       ELSE
         m = A%m
       END IF
     END IF

!  Check for trivial cases

     IF ( trans ) THEN
       IF ( n <= 0 ) THEN
         GO TO 999
       ELSE IF ( m <= 0 ) THEN
         IF ( beta == zero ) THEN
           R( : n ) = zero
         ELSE IF ( beta /= one ) THEN
           R( : n ) = beta * R( : n )
         END IF
         GO TO 999
       END IF
     ELSE
       IF ( m <= 0 ) THEN
         GO TO 999
       ELSE IF ( n <= 0 ) THEN
         IF ( beta == zero ) THEN
           R( : m ) = zero
         ELSE IF ( beta /= one ) THEN
           R( : m ) = beta * R( : m )
         END IF
         GO TO 999
       END IF
     END IF

! Check for bad dimensions.

     IF ( ( m <= 0 ) .OR. ( n <= 0 ) ) THEN
        WRITE( e_dev, 1001 )
        GO TO 999
     END IF

     IF ( symm ) THEN
        IF ( m /= n ) THEN
           WRITE( e_dev, 1002 )
           GO TO 999
        END IF
     END IF

! Print information according to variable printing.

     IF ( printing >= 1 ) THEN

        WRITE( i_dev, "(5X, 'A%type = ', 20a)" ) A%type

        IF ( ALLOCATED( A%id ) ) THEN
           WRITE( i_dev, "(5X, 'A%id   = ', 20a)" ) A%id
        ELSE
           WRITE( i_dev, "(5X, 'A%id   =' )")
        END IF

        WRITE( i_dev, 1008 ) trans, m, alpha,      &
                           symm, n, beta

        IF ( printing >= 2 ) THEN

           SELECT CASE ( SMT_get( A%type ) )

           CASE ( 'DENSE', 'DENSE_BY_COLUMNS' )
              WRITE( i_dev, 1009 ) &
                    ( A%val(i), i = 1, m*n )
           CASE ( 'SPARSE_BY_ROWS' )
              WRITE( i_dev, 1010 ) &
                   ( A%col(i), A%val(i), i=1,A%ptr(m+1)-1 )
              WRITE( i_dev, 1011 ) A%ptr( 1:m+1 )
           CASE ( 'SPARSE_BY_COLUMNS' )
              WRITE( i_dev, 1012 ) ( A%row(i), A%val(i), i = 1, A%ptr(n+1)-1 )
              WRITE( i_dev, 1011 ) A%ptr( 1: n+1 )
           CASE ( 'COORDINATE' )
              WRITE( i_dev, 1013 ) ( A%row(i), A%col(i), A%val(i), i = 1, A%ne )
           CASE ( 'DIAGONAL' )
              WRITE( i_dev, 1009 ) ( A%val(i), i = 1,m )
           CASE( 'SCALED_IDENTITY' )
              WRITE( i_dev, "( /, 5X, ' Identity scaled by ', ES17.10 )" )    &
                A%val(1)
           CASE( 'IDENTITY' )
              WRITE( i_dev, "( /, 5X, ' Identity matrix' )" )
           CASE( 'NONE', 'ZERO' )
              WRITE( i_dev, "( /, 5X, ' Zero matrix' )" )
           CASE DEFAULT
              WRITE( e_dev, 1000 )
              GOTO 999
           END SELECT

           IF ( printing >= 3 ) THEN
              IF ( m == n ) THEN
                 WRITE( i_dev, 1014 ) ( X( i ), R( i ), i = 1, m )
              ELSE IF ( m < n ) THEN
                 WRITE( i_dev, 1014 ) ( X( i ), R( i ), i = 1, m )
                 IF ( trans ) THEN
                   WRITE( i_dev, 1017 ) ( R( i ), i = m + 1, n )
                 ELSE
                   WRITE( i_dev, 1015 ) ( X( i ), i = m + 1, n )
                 END IF
              ELSE
                 WRITE( i_dev, 1014 ) ( X( i ), R( i ), i = 1,n )
                 IF ( trans ) THEN
                   WRITE( i_dev, 1015 ) ( X( i ), i = n + 1, m )
                 ELSE
                   WRITE( i_dev, 1017 ) ( R( i ), i = n + 1, m )
                 END IF
              END IF
              !WRITE( i_dev, 1014 ) X
              !WRITE( i_dev, 1015 ) R
           END IF

        END IF

     END IF

!*****************************************************************
! BEGIN : COMPUTATION                                            !
!*****************************************************************

! First compute beta * R

     IF ( beta == one ) THEN
        ! relax
     ELSE IF ( beta == zero ) THEN
        IF ( trans ) THEN
          R( : n ) = zero
        ELSE
          R( : m ) = zero
        END IF
     ELSE
        IF ( trans ) THEN
           CALL dscal( n, beta, R, 1 )  ! R <- beta * R
        ELSE
           CALL dscal( m, beta, R, 1 )  ! R <- beta * R
        END IF
     END IF

! Compute the rest of the calculation for a given storage type.

     SELECT CASE ( SMT_get( A%type ) )

     ! Storage type GALAHAD_DENSE (by rows)
     ! ************************************

     CASE ( 'DENSE' )

        IF ( symm ) THEN

           nA = 1

           IF ( alpha == one ) THEN

              DO i = 1, m
                 R( i ) = R( i ) + DOT_PRODUCT( A%val( nA : nA+i-1 ), X(1:i) )
                 R( 1 : i-1 ) = R( 1 : i-1 ) + A%val( nA : nA+i-2 ) * X( i )
                 nA = nA + i
              END DO
              !write(*,*) 'Testing 1'

           ELSE

              DO i = 1, m
                 R(i) = R(i) + alpha*DOT_PRODUCT( A%val( nA:nA+i-1 ), X(1:i) )
                 R( 1:i-1 ) = R( 1:i-1 ) + alpha*A%val( nA : nA+i-2 ) * X( i )
                 nA = nA + i
              END DO
              !write(*,*) 'Testing 2'

           END IF
        ELSE
           IF ( trans ) THEN
              IF ( alpha == one ) THEN
                DO j = 1, m
                  R(1:n) = R(1:n) + X(j) * A%val( n*(j-1) + 1: n*j )
                END DO
              ELSE
                DO j = 1, m
                  R(1:n) = R(1:n) + X(j) * alpha * A%val( n*(j-1) + 1 : n*j )
                END DO
             END IF
           ELSE
              IF ( alpha == one ) THEN
                DO j = 1, m
                  R(j) = R(j) + DOT_PRODUCT( A%val( n*(j-1)+1 : n*j ), X(1:n) )
                END DO
              ELSE
                DO j = 1, m
                  R(j) = R(j) + alpha*DOT_PRODUCT(A%val(n*(j-1)+1:n*j), X(1:n))
                END DO
              END IF

           END IF

        END IF


     ! Storage type GALAHAD_DENSE_BY_COLUMNS
     ! *************************************

     CASE ( 'DENSE_BY_COLUMNS' )

        IF ( symm ) THEN

           nA = 1

           IF ( alpha == one ) THEN

             DO j = 1, n
               R( j : m ) = R( j : m ) + A%val( nA : nA+m-j ) * X( j )
               R( j ) = R( j ) + DOT_PRODUCT( A%val( nA+1 : nA+m-j ), X(j+1:m) )
               nA = nA + m - j + 1
             END DO
              !write(*,*) 'Testing 1'

           ELSE

             DO j = 1, n
               R( j : m ) = R( j : m ) + alpha * A%val( nA : nA+m-j ) * X( j )
               R( j ) = R( j ) + alpha *                                      &
                 DOT_PRODUCT( A%val( nA+1 : nA+m-j ), X(j+1:m) )
               nA = nA + m - j + 1
             END DO
              !write(*,*) 'Testing 2'

           END IF
        ELSE
           IF ( trans ) THEN
              IF ( alpha == one ) THEN
                DO j = 1, n
                  R(j) = R(j) + DOT_PRODUCT(A%val(m*(j-1)+1:m*j), X(1:m))
                END DO
              ELSE
                DO j = 1, n
                  R(j) = R(j) + alpha*DOT_PRODUCT(A%val(m*(j-1)+1:m*j), X(1:m))
                END DO
             END IF
           ELSE
              IF ( alpha == one ) THEN
                DO j = 1, n
                  R(1:m) = R(1:m) + A%val( m*(j-1)+1 : m*j ) * X(j)
                END DO
              ELSE
                DO j = 1, n
                  R(1:m) = R(1:m) + alpha*A%val( m*(j-1)+1 : m*j ) * X(j)
                END DO
              END IF

           END IF

        END IF

     ! Storage type GALAHAD_SPARSE_BY_ROWS
     ! ***********************************

     CASE ( 'SPARSE_BY_ROWS' )

        IF ( symm ) THEN

           IF ( alpha == one ) THEN

              DO i = 1, m
                 ri = R( i )
                 Xi = X( i )
                 DO j = A%ptr( i ), A%ptr( i + 1 ) - 1
                    ri = ri + A%val( j ) * X( A%col( j ) )
                    IF ( i /= A%col( j ) ) THEN
                       r( A%col( j ) ) = r( A%col( j ) ) + A%val( j ) * Xi
                    END IF
                 END DO
                 R( i ) = ri
              END DO
              !write(*,*) 'Testing 7'

           ELSE

              DO i = 1, m
                 ri = zero
                 Xi = alpha * X( i )
                 DO j = A%ptr( i ), A%ptr( i + 1 ) - 1
                    ri = ri + A%val( j ) * X( A%col( j ) )
                    IF ( i /= A%col( j ) ) THEN
                       r( A%col(j) ) = r( A%col(j) ) + A%val(j) * Xi
                    END IF
                 END DO
                 R( i ) = R( i ) + alpha * ri
              END DO
              !write(*,*) 'Testing 8'

           END IF

        ELSE

           IF ( trans ) THEN

              IF ( alpha == one ) THEN

                 DO j = 1, m
                    xj = X( j )
                    DO i = A%ptr( j ), A%ptr( j + 1 ) - 1
                       R( A%col( i ) ) = R( A%col( i ) ) + A%val( i ) * xj
                    END DO
                 END DO
                 !write(*,*) 'Testing 9'

              ELSE

                 DO j = 1, m
                    xj = alpha * X( j )
                    DO i = A%ptr( j ), A%ptr( j + 1 ) - 1
                       R( A%col(i) ) = R( A%col(i) ) + A%val(i) * xj
                    END DO
                 END DO
                 !write(*,*) 'Testing 10'

              END IF

           ELSE

              IF ( alpha == one ) THEN

                 DO i = 1, m
                    ri = R( i )
                    DO j = A%ptr( i ), A%ptr( i + 1 ) - 1
                       ri = ri + A%val( j ) * X( A%col( j ) )
                    END DO
                    R( i ) = ri
                 END DO
                 !write(*,*) 'Testing 11'

              ELSE

                 DO i = 1, m
                    ri = zero
                    DO j = A%ptr( i ), A%ptr( i + 1 ) - 1
                       ri = ri + A%val( j ) * X( A%col( j ) )
                    END DO
                    R( i ) = R( i ) + alpha * ri
                 END DO
                 !write(*,*) 'Testing 12'

              END IF

           END IF

        END IF

     ! Storage type GALAHAD_SPARSE_BY_COLUMNS
     ! **************************************

     CASE ( 'SPARSE_BY_COLUMNS' )

        IF ( symm ) THEN

           IF ( alpha == one ) THEN

              DO j = 1, n
                 rj = R( j )
                 Xj = X( j )
                 DO i = A%ptr( j ), A%ptr( j + 1 ) - 1
                    R( A%row( i ) ) = R( A%row( i ) ) + A%val( i ) * Xj
                    IF ( j /= A%row( i ) ) THEN
                       R( j ) = R( j ) + A%val( i ) * X( A%row( i ) )
                    END IF
                 END DO
              END DO
              !write(*,*) 'Testing 13'

           ELSE

              DO j = 1, n
                 Xj = alpha * X( j )
                 DO i = A%ptr( j ), A%ptr( j + 1 ) - 1
                    R( A%row( i ) ) = R( A%row( i ) ) + A%val( i ) * Xj
                    IF ( j /= A%row( i ) ) THEN
                       R( j ) = R( j ) + alpha*A%val( i ) * X( A%row( i ) )
                    END IF
                 END DO
              END DO
              !write(*,*) 'Testing 14'

           END IF

        ELSE
           IF ( trans ) THEN

              IF ( alpha == one ) THEN

                 DO j = 1, n
                    rj = R( j )
                    DO i = A%ptr( j ), A%ptr( j + 1 ) - 1
                       rj = rj + A%val( i ) * X( A%row( i ) )
                    END DO
                    R( j ) = rj
                 END DO
                 !write(*,*) 'Testing 15'

              ELSE

                 DO j = 1, n
                    rj = zero
                    DO i = A%ptr( j ), A%ptr( j + 1 ) - 1
                       rj = rj + A%val( i ) * X( A%row( i ) )
                    END DO
                    R( j ) = R( j ) + alpha * rj
                 END DO
                 !write(*,*) 'Testing 16'

              END IF


           ELSE

              IF ( alpha == one ) THEN

                 DO j = 1, n
                    Xj = X( j )
                    DO i = A%ptr( j ), A%ptr( j + 1 ) - 1
                       R( A%row( i ) ) = R( A%row( i ) ) + A%val( i ) * Xj
                    END DO
                 END DO
                 !write(*,*) 'Testing 17'

              ELSE

                 DO j = 1, n
                    Xj = alpha * X( j )
                    DO i = A%ptr( j ), A%ptr( j + 1 ) - 1
                       R( A%row(i) ) = R( A%row(i) ) + A%val(i)*Xj
                    END DO
                 END DO
                 !write(*,*) 'Testing 18'

              END IF

           END IF

        END IF

     ! Storage type GALAHAD_COORDINATE
     ! *******************************

     CASE ( 'COORDINATE' )

        IF ( symm ) THEN

           IF ( alpha == one ) THEN

              DO i = 1, A%ne

                 Arowi = A%row(i)
                 Acoli = A%col(i)
                 Avali = A%val(i)

                 R( Arowi ) = R( Arowi ) + Avali * X( Acoli )

                 IF ( Arowi /= Acoli ) THEN
                    R( Acoli ) = R( Acoli ) + Avali * X( Arowi )
                 END IF

              END DO
              !WRITE(*,*) 'Testing 19'

           ELSE

              DO i = 1, A%ne

                 Avali = alpha * A%val(i)
                 Arowi = A%row(i)
                 Acoli = A%col(i)

                 R( Arowi ) = R( Arowi ) + Avali * X( Acoli )

                 IF ( Arowi /= Acoli ) THEN
                    R( Acoli ) = R( Acoli ) + Avali * X( Arowi )
                 END IF

              END DO
              !WRITE(*,*) 'Testing 20'

           END IF
        ELSE
          IF ( trans ) THEN
            IF ( alpha == one ) THEN
              DO i = 1, A%ne
                R( A%col(i) ) = R( A%col(i) ) + A%val(i) * X( A%row(i) )
              END DO
            ELSE
              DO i = 1, A%ne
                R( A%col(i) ) = R( A%col(i) ) + alpha*A%val(i) * X( A%row(i) )
              END DO
            END IF
          ELSE
            IF ( alpha == one ) THEN
              DO i = 1, A%ne
                R( A%row(i) ) = R( A%row(i) ) + A%val(i) * X( A%col(i) )
              END DO
            ELSE
              DO i = 1, A%ne
                R( A%row(i) ) = R( A%row(i) ) + alpha*A%val(i) * X( A%col(i) )
              END DO
            END IF
          END IF
        END IF

     ! Storage type GALAHAD_DIAGONAL
     ! *****************************

     CASE( 'DIAGONAL' )

        IF ( alpha == one ) THEN

           R( 1:n ) = R( 1:n ) + A%val( 1:n ) * X( 1:n )
           !WRITE(*,*) 'Testing 25'

        ELSE

           R( 1:n ) = R( 1:n ) + alpha*A%val( 1:n ) * X( 1:n )
           !WRITE(*,*) 'Testing 26'

        END IF

     CASE( 'SCALED_IDENTITY' )

        IF ( alpha == one ) THEN

           R( 1:n ) = R( 1:n ) + A%val( 1 ) * X( 1:n )
           !WRITE(*,*) 'Testing 25'

        ELSE

           R( 1:n ) = R( 1:n ) + alpha*A%val( 1 ) * X( 1:n )
           !WRITE(*,*) 'Testing 26'

        END IF

     CASE( 'IDENTITY' )

        IF ( alpha == one ) THEN

           R( 1:n ) = R( 1:n ) + X( 1:n )
           !WRITE(*,*) 'Testing 25'

        ELSE

           R( 1:n ) = R( 1:n ) + alpha * X( 1:n )
           !WRITE(*,*) 'Testing 26'

        END IF

     CASE( 'NONE', 'ZERO' )

    ! Invalid storage type given in A%type
    ! ************************************

     CASE DEFAULT

           WRITE( e_dev, 1000 )

     END SELECT

!*****************************************************************
! END : COMPUTATION                                              !
!*****************************************************************

999 CONTINUE

     ! Print final solution, if necessary

     IF ( printing >= 1 ) THEN
        IF ( printing >= 3 ) THEN
           WRITE( i_dev, 1016 ) R
        END IF
        WRITE( i_dev, 1020 ) ! footer
     END IF

     RETURN

!*****************************************************************

!  Print formats

1000  FORMAT(/,5X,'*** ERROR : mop_AX : Unrecognized value A%type.')
1001  FORMAT(/,5X,'*** ERROR : mop_AX : A%m <= 0 and/or A%n <= 0 .')
1002  FORMAT(/,5X,'*** ERROR : mop_AX : '                           ,          &
                  'symmetric = .TRUE., but A%m /= A%n.',/)
1005  FORMAT(/,                                                                &
      3X,'*************************************************************',/,    &
      3X,'*                     BEGIN: mop_Ax                         *',/,    &
      3X,'*        GALAHAD sparse matrix operation subroutine         *',/,    &
      3X,'*************************************************************',/)
1008  FORMAT(/,                                                                &
      5X,'transpose = ', L1, 5X, 'm =', I6, 5X,'alpha =', ES17.10,/,           &
      5X,'symmetric = ', L1, 5X, 'n =', I6, 5X,'beta  =', ES17.10 )
1009  FORMAT(/,5X,'      A%val    ',/, &
               5X,'  -------------',/, (5X, ES17.10 ) )
1010  FORMAT(/,5X,'  A%col             A%val    ',/,  &
               5X,'  -----         -------------',/,  &
              (5X, I7, 7X, ES17.10) )
1011  FORMAT(/,5X,'  A%ptr',/, &
               5X,'  -----',/, (5X, I7 ) )
1012  FORMAT(/,5X,'  A%row             A%val    ',/,  &
               5X,'  -----         -------------',/,  &
              (5X, I7, 7X, ES17.10) )
1013  FORMAT(/,5X,'  A%row         A%col             A%val    ',/,  &
               5X,'  -----         -----         -------------',/,  &
              (5X, I7, 7X, I7, 7X, ES17.10) )
1014  FORMAT(/,5X,'         X                   R(in)', /,  &
               5X,'     ---------             ---------',/,  &
              (5X, ES17.10, 5X, ES17.10) )
1015  FORMAT( (5X, ES17.10) )
1016  FORMAT(/,5X,'     R (out)   ',/,  &
               5X,'  -------------',/,  &
              (5X, ES17.10) )
1017  FORMAT( (27X, ES17.10) )
1020  FORMAT(/,                                                                &
      3X,'*************************************************************',/,    &
      3X,'*                      END: mop_Ax                          *',/,    &
      3X,'*************************************************************',/)

   END SUBROUTINE mop_Ax

!-*-*-*-*-*-* - E N D : m o p _ A X   S U B R O U T I N E  -*-*-*-*-*-*-*-*-*-

!-*-*-*-*-  B E G I N  m o p _ s c a l e A  S U B R O U T I N E  -*-*-*-*-*-*-

   SUBROUTINE mop_scaleA( A, u, v, out, error, print_level, symmetric )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................................
!      .                                                            .
!      .  Scales row i of A by u_i and column j of A by v_j.        .
!      .  If A is symmetric, it scales rows and columns by u.       .
!      .                                                            .
!      ..............................................................
!
!  Arguments:
!  =========
!
!   A        a scalar variable of derived type SMT_type.
!
!   out      (optional) is a scalar variable of type integer, which
!            holds the stream number for informational messages;  the
!            file is assumed to already have been opened.  Default
!            is out = 6.
!
!   error    (optional) is a scalar variable of type integer, which
!            holds the stream number for error messages; the file
!            is assumed to already have been opened.  Default is error = 6.
!
!   print_level  (optional) is scalar variable of type integer.  It
!                controls the amount of printed information to unit
!                number defined by out.  Values and their meanings are:
!
!                print_level <= 0    Nothing.
!                print_level  = 1    The following is printed: A%m, A%n,
!                                    A%type, A%id, A%row, A%col, A%val,
!                                    A%ptr, u, and v.
!
!   symmetric (optional)
!            is a scalar variable of type logical.  If
!            symmetric = .TRUE., then the matrix A is assumed symmetric.
!            If symmetric = .FALSE. or is not present, then the matrix
!            A is assumed to NOT be symmetric.
!
!   u        (optional) vector variable of length A%m.  Holds the values for
!            which the A%m rows will be scaled.
!
!   v        (optional) a vector variable of length A%n.  Holds the values for
!            which the A%n columns will be scaled.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

! Dummy arguments

  TYPE( SMT_type), INTENT( INOUT  ) :: A
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( : ), OPTIONAL :: u, v
  INTEGER, INTENT( IN ), OPTIONAL :: error, out, print_level
  LOGICAL, INTENT( IN ), OPTIONAL :: symmetric

!  Local variables

  INTEGER :: e_dev, i_dev, printing
  INTEGER :: i, j, l
  INTEGER :: m, n, first, last, element
  REAL ( KIND = wp ) :: ui, uj, vj
  LOGICAL :: symm

! Determine values for optional parameters.

  IF ( PRESENT( error ) ) THEN
     e_dev = error
  ELSE
     e_dev = 6
  END IF

  IF ( PRESENT( out ) ) THEN
     i_dev = out
  ELSE
     i_dev = 6
  END IF

  IF ( PRESENT( print_level ) ) THEN
     printing = print_level
  ELSE
     printing = 0
  END IF

  IF ( PRESENT( symmetric ) ) THEN
     symm = symmetric
  ELSE
     symm = .false.
  END IF

! Print header

  IF ( printing >= 1 ) WRITE( i_dev, 1006 )

! Make sure that at least one of u and v is supplied.

  if ( .not. present( u ) .and. .not. present( v ) ) then
     write( e_dev, 1000 )
     return
  end if

! Make sure that symmetric matrix case makes sense.

  if ( symm ) then
     if ( .not. present(u) ) then
        write( e_dev, 1016 )
        goto 999
     end if
  end if

! Set some convenient variables.

  m = A%m
  n = A%n

! Check for bad dimensions.

  IF ( ( m <= 0 ) .OR. ( n <= 0 ) ) THEN
     WRITE( e_dev, 1001 )
     GOTO 999
  END IF

  IF ( present(u) ) then
     if ( size(u) /= m ) then
        WRITE( e_dev, 1002 )
        GOTO 999
     end if
  END IF
  IF ( present(v) ) then
     if ( size(v) /= n ) then
        WRITE( e_dev, 1002 )
        GOTO 999
     end if
  END IF

  IF ( symm .and. m /= n ) then
     WRITE( e_dev, 1007 )
     GOTO 999
  END IF

! Print information according to variable printing.

  IF ( printing >= 1 ) THEN

     WRITE( i_dev, 1004 )

     WRITE( i_dev, "(5X, 'A%type    = ', 20a)" ) A%type

     IF ( ALLOCATED( A%id ) ) THEN
        WRITE( i_dev, "(5X, 'A%id      = ', 20a)" ) A%id
     ELSE
        WRITE( i_dev, "(5X, 'A%id      =' )")
     END IF

     WRITE( i_dev, "(5X, 'SYMMETRIC = ', L1)" ) symm

     WRITE( i_dev, 1008 ) m, n

     SELECT CASE ( SMT_get( A%type ) )

     CASE ( 'DENSE' )
        WRITE( i_dev, 1009 ) &
             ( A%val(i), i = 1, m*n )
     CASE ( 'SPARSE_BY_ROWS' )
        WRITE( i_dev, 1010 ) &
             ( A%col(i), A%val(i), i=1,A%ptr(m+1)-1 )
        WRITE( i_dev, 1011 ) A%ptr( 1:m+1 )
     CASE ( 'SPARSE_BY_COLUMNS' )
        WRITE( i_dev, 1012 ) &
             ( A%row(i), A%val(i), &
             i = 1, A%ptr(n+1)-1 )
        WRITE( i_dev, 1011 ) A%ptr( 1: n+1 )
     CASE ( 'COORDINATE' )
        WRITE( i_dev, 1013 ) &
             ( A%row(i), A%col(i), A%val(i),  &
             i = 1,A%ne )
     CASE ( 'DIAGONAL' )
        WRITE( i_dev, 1009 ) &
             ( A%val(i), i = 1,m )
     CASE DEFAULT
        WRITE( e_dev, 1003 )
        GOTO 999
     END SELECT

     IF ( .not. present(u) ) then
        WRITE( i_dev, 1015 ) 'v', ( v(i), i = 1,n )
     ELSEIF ( .not. present(v) ) then
        WRITE( i_dev, 1015 ) 'u', ( u(i), i = 1,m )
     ELSE
        IF ( m == n ) THEN
           WRITE( i_dev, 1014 ) ( u(i), v(i), i = 1,m )
        ELSEIF ( m < n ) THEN
           WRITE( i_dev, 1014 ) ( u(i), v(i), i = 1,m )
           WRITE( i_dev, 1017 ) ( v(i), i = m+1,n )
        ELSE
           WRITE( i_dev, 1014 ) ( u(i), v(i), i = 1,n )
           WRITE( i_dev, 1018 ) ( u(i), i = n+1,m )
        END IF
     END IF

  END IF

!*****************************************************************
! BEGIN : COMPUTATION                                            !
!*****************************************************************

  if ( symm ) then

     ! ------------------------------------------
     ! Hit A on the left and right side with u. |
     ! ------------------------------------------

     SELECT CASE ( SMT_get( A%type ) )

     CASE ( 'DENSE' )

        element = 1
        do i = 1, m
           ui = u(i)
           do j = 1, i
              A%val( element ) = ui * A%val( element ) * u(j)
              element = element + 1
           end do
        end do

     CASE ( 'SPARSE_BY_ROWS' )

        do i = 1, m
           ui = u(i)
           do l = A%ptr(i), A%ptr(i+1)-1
              A%val(l) = ui * A%val(l) * u( A%col(l) )
           end do
        end do

     CASE ( 'SPARSE_BY_COLUMNS' )

        do j = 1, n
           uj = u(j)
           do l = A%ptr(j), A%ptr(j+1)-1
              A%val(l) = uj * A%val(l) * u( A%row(l) )
           end do
        end do

     CASE ( 'COORDINATE' )

        do i = 1, A%ne
           A%val(i) = u( A%row(i) ) * A%val(i) * u( A%col(i) )
        end do

     CASE ( 'DIAGONAL' )

        do i = 1, n
           A%val(i) = u(i)**2 * A%val(i)
        end do

     CASE DEFAULT

        WRITE( e_dev, 1003 )

     END SELECT

  else   ! not symmetric.

     if ( present(u) .and. present(v) ) then

        ! --------------------------------------------------------------
        ! Hit A on the left and right side with u and v, respectively. |
        ! --------------------------------------------------------------

        SELECT CASE ( SMT_get( A%type ) )

        CASE ( 'DENSE' )

           element = 1
           do i = 1, m
              ui = u(i)
              do j = 1, n
                 A%val( element ) = ui * A%val( element ) * v(j)
                 element = element + 1
              end do
           end do

        CASE ( 'SPARSE_BY_ROWS' )

           do i = 1, m
              ui = u(i)
              do l = A%ptr(i), A%ptr(i+1)-1
                 A%val(l) = ui * A%val(l) * v( A%col(l) )
              end do
           end do

        CASE ( 'SPARSE_BY_COLUMNS' )

           do j = 1, n
              vj = v(j)
              do l = A%ptr(j), A%ptr(j+1)-1
                 A%val(l) = u( A%row(l) ) * A%val(l) * vj
              end do
           end do

        CASE ( 'COORDINATE' )

           do i = 1, A%ne
              A%val(i) = u( A%row(i) ) * A%val(i) * v( A%col(i) )
           end do

        CASE ( 'DIAGONAL' )

           do i = 1, n
              A%val(i) = u(i) * A%val(i) * v(i)
           end do

        CASE DEFAULT

           WRITE( e_dev, 1003 )

        END SELECT

     elseif ( present(u) ) then

        ! -------------------------------------
        ! Only hit A on the left side with u. |
        ! -------------------------------------

        SELECT CASE ( SMT_get( A%type ) )

        CASE ( 'DENSE' )

           if ( symm ) then
              first = 1
              do i = 1, m
                 ui = u(i)
                 last = first + i - 1
                 A%val( first:last ) = ui * A%val( first:last )
                 first = first + i
              end do
           else
              first = 1
              do i = 1, m
                 ui = u(i)
                 last = first + n - 1
                 A%val( first:last ) = ui * A%val( first:last )
                 first = first + n
              end do
           end if

        CASE ( 'SPARSE_BY_ROWS' )

           do i = 1, m
              ui = u(i)
              first = A%ptr(i)
              last = A%ptr(i+1) - 1
              A%val( first:last ) = ui * A%val( first:last )
           end do

        CASE ( 'SPARSE_BY_COLUMNS' )

           do j = 1, n
              do l = A%ptr(j), A%ptr(j+1) - 1
                 A%val( l ) = u( A%row(l) ) * A%val( l )
              end do
           end do

        CASE ( 'COORDINATE' )

           do i = 1, A%ne
              A%val(i) = u( A%row(i) ) * A%val(i)
           end do

        CASE ( 'DIAGONAL' )

           do i = 1, n
              A%val(i) = u(i) * A%val(i)
           end do

        CASE DEFAULT

           WRITE( e_dev, 1003 )

        END SELECT

     else

        ! --------------------------------------
        ! Only hit A on the right side with v. |
        ! --------------------------------------

        SELECT CASE ( SMT_get( A%type ) )

        CASE ( 'DENSE' )

           if ( symm ) then
              first = 0
              do i = 1, m
                 first = first + i - 1
                 do j = 1, i
                    A%val( first + j ) = A%val( first + j ) * v( j )
                 end do
              end do
           else
              do i = 1, m
                 first = (i-1)*n
                 do j = 1, n
                    A%val( first + j ) = A%val( first + j ) * v( j )
                 end do
              end do
           end if

        CASE ( 'SPARSE_BY_ROWS' )

           do i = 1, m
              do l = A%ptr(i), A%ptr(i+1) - 1
                 A%val( l ) = A%val( l ) * v( A%col(l) )
              end do
           end do

        CASE ( 'SPARSE_BY_COLUMNS' )

           do j = 1, n
              vj = v(j)
              first = A%ptr(j)
              last = A%ptr(j+1) - 1
              A%val( first:last ) = vj * A%val( first:last )
           end do

        CASE ( 'COORDINATE' )

           do i = 1, A%ne
              A%val(i) = A%val(i) * v( A%col(i) )
           end do

        CASE ( 'DIAGONAL' )

           do i = 1, n
              A%val(i) = A%val(i) * v(i)
           end do

        CASE DEFAULT

           WRITE( e_dev, 1003 )

        END SELECT

     end if

  end if

!*****************************************************************
! END : COMPUTATION                                              !
!*****************************************************************

  ! Print matrix values after the scaling.

  IF ( printing >= 1 ) THEN

     WRITE( i_dev, 1005 )

     SELECT CASE ( SMT_get( A%type ) )

     CASE ( 'DENSE' )
        WRITE( i_dev, 1009 ) &
             ( A%val(i), i = 1, m*n )
     CASE ( 'SPARSE_BY_ROWS' )
        WRITE( i_dev, 1010 ) &
             ( A%col(i), A%val(i), i=1,A%ptr(m+1)-1 )
        WRITE( i_dev, 1011 ) A%ptr( 1:m+1 )
     CASE ( 'SPARSE_BY_COLUMNS' )
        WRITE( i_dev, 1012 ) &
             ( A%row(i), A%val(i), &
             i = 1, A%ptr(n+1)-1 )
        WRITE( i_dev, 1011 ) A%ptr( 1: n+1 )
     CASE ( 'COORDINATE' )
        WRITE( i_dev, 1013 ) &
             ( A%row(i), A%col(i), A%val(i),  &
             i = 1,A%ne )
     CASE ( 'DIAGONAL' )
        WRITE( i_dev, 1009 ) &
             ( A%val(i), i = 1,m )
     CASE DEFAULT
        WRITE( e_dev, 1003 )
        GOTO 999
     END SELECT

  END IF

999 CONTINUE

! Print footer.

  IF ( printing >= 1 ) THEN
     WRITE( i_dev, 1020 ) ! footer
  END IF

  RETURN

! format statements

1000 FORMAT(/,5X,'*** ERROR : mop_scaleA : must supply at ', &
                 'least one of u and v.')
1001  FORMAT(/,5X,'*** ERROR : mop_scaleA : A%m <= 0 and/or A%n <= 0 .')
1002  FORMAT(/,5X,'*** ERROR : mop_scaleA : invalid length for u and/or v.')
1003  FORMAT(/,5X,'*** ERROR : mop_scaleA : Unrecognized value A%type.')
1004  FORMAT(                                                                  &
      10X,'   ----------------------------',/,                                 &
      10X,'        Matrix Pre-scaling',/,                                      &
      10X,'   ----------------------------',/ )
1005  FORMAT(/,                                                                &
      10X,'   -----------------------------',/,                                &
      10X,'        Matrix Post-scaling',/,                                     &
      10X,'   -----------------------------' )
1006  FORMAT(                                                                  &
      3X,'*************************************************************',/,    &
      3X,'*                 BEGIN: mop_scaleA                         *',/,    &
      3X,'*        GALAHAD sparse matrix operation subroutine         *',/,    &
      3X,'*************************************************************',/)
1007  FORMAT(/,5X,'*** ERROR : mop_scaleA : entered as symmetric, but m/=n.')
1008  FORMAT(/, 5X, '(m,n) = (', I6, ',', I6, ')' )
1009  FORMAT(/,5X,'      A%val    ',/, &
               5X,'  -------------',/, (5X, ES17.10 ) )
1010  FORMAT(/,5X,'  A%col             A%val    ',/,  &
               5X,'  -----         -------------',/,  &
              (5X, I7, 7X, ES17.10) )
1011  FORMAT(/,5X,'  A%ptr',/, &
               5X,'  -----',/, (5X, I7 ) )
1012  FORMAT(/,5X,'  A%row             A%val    ',/,  &
               5X,'  -----         -------------',/,  &
              (5X, I7, 7X, ES17.10) )
1013  FORMAT(/,5X,'  A%row         A%col             A%val    ',/,  &
               5X,'  -----         -----         -------------',/,  &
              (5X, I7, 7X, I7, 7X, ES17.10) )
1014  FORMAT(/,5X,'         u                     v', /,  &
               5X,'     ---------             ---------',/,  &
              (5X, ES17.10, 5X, ES17.10) )
1015  FORMAT(/,18X,'         ', 1A, /, 18X,'     ---------',/, (18X, ES17.10) )
1016  FORMAT(/,5X,'*** ERROR : mop_scaleA : A symmetric, must supply u.')
1017  FORMAT( (27X, ES17.10) )
1018  FORMAT( ( 5X, ES17.10) )
1020  FORMAT(/,                                                                &
      3X,'*************************************************************',/,    &
      3X,'*                  END: mop_scaleA                          *',/,    &
      3X,'*************************************************************',/)

   end SUBROUTINE mop_scaleA

!-*-*-*-*  B E G I N  m o p _ g e t v a l   S U B R O U T I N E  *-*-*-*-*-*

   SUBROUTINE mop_getval( A, row, col, val, symmetric, out, error, print_level )
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ................................................................
!      .                                                              .
!      .      Returns the value of the (i,j) entry of matrix A.       .
!      .                                                              .
!      ................................................................

!  Arguments:
!  =========

!   A        a scalar variable of derived type SMT_type.
!
!   row,col  scalar variables of type integer.  References the (i,j)th
!            entry that the user wishes to be returned.
!
!   val      scalar variable of type real.  On successful exit contains
!            the value of the (i,j)th entry of the matrix A.
!
!   symmetric (optional) is a scalar variable of type logical.  Set
!             symmetric = .TRUE. if the matrix A is symmetric; otherwise
!             set symmetric = .FALSE.  If not present, then the
!             value symmetric = .FALSE. is assumed.
!
!   out      (optional) is a scalar variable of type integer, which
!            holds the stream number for informational messages;  the
!            file is assumed to already have been opened.  Default
!            is out = 6.
!
!   error    (optional) is a scalar variable of type integer, which
!            holds the stream number for error messages; the file
!            is assumed to already have been opened.  Default is error = 6.
!
!   print_level  (optional) is scalar variable of type integer.  It
!                controls the amount of printed information to unit
!                number defined by out.  Values and their meanings are:
!
!                print_level <= 0    Nothing.
!                print_level  = 1    The following is printed: A%m, A%n,
!                                    A%type, A%id, row, col, and val.
!                print_level  = 2    Those from above, in additon to:
!                                    A%ptr, A%val, A%row, and A%col.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     INTEGER, INTENT( IN ) :: row, col
     TYPE( SMT_type), INTENT( IN  ) :: A
     REAL( KIND = wp ), INTENT( OUT ) :: val
     LOGICAL, INTENT( IN ), OPTIONAL :: symmetric
     INTEGER, INTENT( IN ), OPTIONAL :: error, out, print_level

! Set Parameters

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one  = 1.0_wp

!  Local variables

     INTEGER :: ii, i, j, m, n
     INTEGER :: i_dev, e_dev, printing
     LOGICAL :: symm

!**************************************************************

! Check for optional arguments.

     if ( present( symmetric ) ) then
        symm = symmetric
     else
        symm = .false.
     end if

    if ( present( error ) ) then
       e_dev = error
    else
       e_dev = 6
    end if

    if ( present( out ) ) then
       i_dev = out
    else
       i_dev = 6
    end if

    if ( present( print_level ) ) then
       printing = print_level
    else
       printing = 0
    end if

! Print Header

     IF ( printing >= 1 ) WRITE( i_dev, 1005 )

! Set some convenient variables.

     m = A%m
     n = A%n

! Check for bad dimensions.

     IF ( ( m <= 0 ) .OR. ( n <= 0 ) ) THEN
        WRITE( e_dev, 1001 )
        GOTO 999
     END IF

     IF ( symm ) THEN
        IF ( m /= n ) THEN
           WRITE( e_dev, 1002 )
           GOTO 999
        END IF
     END IF

! Print information according to variable printing.

     IF ( printing >= 1 ) THEN

        WRITE( i_dev, "(5X, 'A%type = ', 20a)" ) A%type

        IF ( ALLOCATED( A%id ) ) THEN
           WRITE( i_dev, "(5X, 'A%id   = ', 20a)" ) A%id
        ELSE
           WRITE( i_dev, "(5X, 'A%id   =' )")
        END IF

        WRITE( i_dev, 1008 ) m, row, symm, n, col

        IF ( printing >= 2 ) THEN

           SELECT CASE ( SMT_get( A%type ) )

           CASE ( 'DENSE' )
              WRITE( i_dev, 1009 ) &
                    ( A%val(i), i = 1, m*n )
           CASE ( 'SPARSE_BY_ROWS' )
              WRITE( i_dev, 1010 ) &
                   ( A%col(i), A%val(i), i=1,A%ptr(m+1)-1 )
              WRITE( i_dev, 1011 ) A%ptr( 1:m+1 )
           CASE ( 'SPARSE_BY_COLUMNS' )
              WRITE( i_dev, 1012 ) &
                   ( A%row(i), A%val(i), &
                   i = 1, A%ptr(n+1)-1 )
              WRITE( i_dev, 1011 ) A%ptr( 1: n+1 )
           CASE ( 'COORDINATE' )
              WRITE( i_dev, 1013 ) &
                   ( A%row(i), A%col(i), A%val(i),  &
                   i = 1,A%ne )
           CASE ( 'DIAGONAL' )
              WRITE( i_dev, 1009 ) &
                   ( A%val(i), i = 1,m )
           CASE DEFAULT
              WRITE( e_dev, 1000 )
              GOTO 999
           END SELECT

        END IF

     END IF

! Account for possible symmetric storage of the matrices.

     if ( symm ) then
        if ( row < col ) then
           i = col
           j = row
        else
           i = row
           j = col
        end if
     else
        i = row
        j = col
     end if

 ! ------------------------------------------
 ! Get correct value based on storage type. -
 ! ------------------------------------------

 SELECT CASE ( SMT_get( A%type ) )

     CASE ( 'DENSE' )
     !***************

        if ( symm ) then
           val = A%val( (i*(i-1))/2 + j )  ! DPR: smart way?
        else
           val = A%val( n * (i-1) + j )
        end if

     CASE ( 'SPARSE_BY_ROWS' )
     !************************

        val = zero
        do ii = A%ptr(i), A%ptr(i+1)-1
           if ( A%col(ii) == j ) then
              val = A%val( ii )
              EXIT
           end if
        end do

     CASE ( 'SPARSE_BY_COLUMNS' )
     !***************************

        val = zero
        do ii = A%ptr(j), A%ptr(j+1)-1
           if ( A%row(ii) == i ) then
              val = A%val( ii )
              EXIT
           end if
        end do

     CASE ( 'COORDINATE' )
     !********************

        val = zero
        do ii = 1, A%ne
           if ( A%row(ii) == i ) then
              if ( A%col(ii) == j ) then
                 val = A%val(ii)
                 EXIT
              end if
           end if
        end do

     CASE ( 'DIAGONAL' )
     !******************

        if ( i == j ) then
           val = A%val(i)
        else
           val = zero
        end if

    CASE ( 'SCALED_IDENTITY' )
    !******************

        if ( i == j ) then
           val = A%val(1)
        else
           val = zero
        end if


    CASE ( 'IDENTITY' )
    !******************

        if ( i == j ) then
           val = one
        else
           val = zero
        end if


    CASE( 'NONE', 'ZERO' )
    !******************

         val = zero

     CASE DEFAULT
     !***********

        val = zero
        write( e_dev, 1000 )

     END SELECT

999  CONTINUE

     ! possibly print some more.

     IF ( printing >= 1 ) THEN
        WRITE( i_dev, 1014 ) val
        WRITE( i_dev, 1020 ) ! footer
     END IF

     RETURN

!  Format statements

1000  FORMAT(/,5X,'*** ERROR : mop_getval : Unrecognized value A%type.')
1001  FORMAT(/,5X,'*** ERROR : mop_getval : A%m <= 0 and/or A%n <= 0 .')
1002  FORMAT(/,5X,'*** ERROR : mop_getval : ',                                 &
                  'symmetric = .TRUE., but A%m /= A%n.',/)
1005  FORMAT(/,                                                                &
      3X,'*************************************************************',/,    &
      3X,'*                     BEGIN: mop_getval                     *',/,    &
      3X,'*      GALAHAD gets a single element of a sparse matrix     *',/,    &
      3X,'*************************************************************',/)
1008  FORMAT(/,                                                                &
      5X,'m =', I6, 5X,'row =', I6, 5X, 'symmetric = ', L1, /,                 &
      5X,'n =', I6, 5X,'col =', I6 )
1009  FORMAT(/,5X,'      A%val    ',/, &
               5X,'  -------------',/, (5X, ES17.10 ) )
1010  FORMAT(/,5X,'  A%col             A%val    ',/,  &
               5X,'  -----         -------------',/,  &
              (5X, I7, 7X, ES17.10) )
1011  FORMAT(/,5X,'  A%ptr',/, &
               5X,'  -----',/, (5X, I7 ) )
1012  FORMAT(/,5X,'  A%row             A%val    ',/,  &
               5X,'  -----         -------------',/,  &
              (5X, I7, 7X, ES17.10) )
1013  FORMAT(/,5X,'  A%row         A%col             A%val    ',/,             &
               5X,'  -----         -----         -------------',/,             &
              (5X, I7, 7X, I7, 7X, ES17.10) )
1014  FORMAT(/,5X,'ON EXIT: value = ', ES16.9 )
1020  FORMAT(/,                                                                &
      3X,'*************************************************************',/,    &
      3X,'*                      END: mop_getval                      *',/,    &
      3X,'*************************************************************',/)

   END SUBROUTINE mop_getval

!-*-*-*-*  E N D : m o p _  g e t v a l   S U B R O U T I N E  *-*-*-*-*-*

!-*-*-  B E G I N  m o p _ r o w _ 2 _ n o r m s   S U B R O U T I N E  *-*-*-

   SUBROUTINE mop_row_2_norms( A, row_norms, symmetric, out, error,            &
                               print_level )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................................
!      .                                                            .
!      .  Returns the vector of row-wise two norms of the matrix A. .
!      .                                                            .
!      ..............................................................

!  Arguments:
!  =========

!   A        a scalar variable of derived type SMT_type.
!
!   row_norms  rank 1 array of type real.  The value row_norms(i) gives the
!              2-norm of the i-th row.
!
!   symmetric (optional) is a scalar variable of type logical.  Set
!             symmetric = .TRUE. if the matrix A is symmetric; otherwise
!             set symmetric = .FALSE.  If not present, then the
!             value symmetric = .FALSE. is assumed.
!
!   out      (optional) is a scalar variable of type integer, which
!            holds the stream number for informational messages;  the
!            file is assumed to already have been opened.  Default
!            is out = 6.
!
!   error    (optional) is a scalar variable of type integer, which
!            holds the stream number for error messages; the file
!            is assumed to already have been opened.  Default is error = 6.
!
!   print_level  (optional) is scalar variable of type integer.  It
!                controls the amount of printed information to unit
!                number defined by out.  Values and their meanings are:
!
!                print_level <= 0    Nothing.
!                print_level  = 1    The following is printed: A%m, A%n,
!                                    A%type, A%id, row, col, and val.
!                print_level  = 2    Those from above, in additon to:
!                                    A%ptr, A%val, A%row, and A%col.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

  TYPE( SMT_type), INTENT( IN  ) :: A
  REAL( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: row_norms
  LOGICAL, INTENT( IN ), OPTIONAL :: symmetric
  INTEGER, INTENT( IN ), OPTIONAL :: error, out, print_level

! Set Parameters

  REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
  REAL ( KIND = wp ), PARAMETER :: one  = 1.0_wp

!  Local variables

  INTEGER :: i, j, l, l1, l2, m, n
  INTEGER :: i_dev, e_dev, printing
  REAL ( KIND = wp ) :: val
  LOGICAL :: symm

!  Check for optional arguments

  IF ( PRESENT( symmetric ) ) THEN
    symm = symmetric
  ELSE
    symm = .false.
  END IF
  IF ( PRESENT( error ) ) THEN
    e_dev = error
  ELSE
    e_dev = 6
  END if
  IF ( PRESENT( out ) ) THEN
    i_dev = out
  ELSE
    i_dev = 6
  END IF
    IF ( PRESENT( print_level ) ) THEN
    printing = print_level
  ELSE
    printing = 0
  END IF

!   Print Header

  IF ( printing >= 1 ) WRITE( i_dev, 1005 )

!   Set some convenient variables

  m = A%m ; n = A%n

!   Check for bad dimensions

  IF ( ( m <= 0 ) .OR. ( n <= 0 ) ) THEN
    WRITE( e_dev, 1001 )
    GOTO 999
  END IF

  IF ( symm ) THEN
    IF ( m /= n ) THEN
      WRITE( e_dev, 1002 )
      GOTO 999
    END IF
  END IF

!  Print information according to variable printing

  IF ( printing >= 1 ) THEN
    WRITE( i_dev, "(5X, 'A%type = ', 20a)" ) A%type
    IF ( ALLOCATED( A%id ) ) THEN
       WRITE( i_dev, "(5X, 'A%id   = ', 20a)" ) A%id
    ELSE
       WRITE( i_dev, "(5X, 'A%id   =' )")
    END IF

    WRITE( i_dev, 1008 ) m, n, symm
    IF ( printing >= 2 ) THEN
      SELECT CASE ( SMT_get( A%type ) )
      CASE ( 'DENSE' )
        WRITE( i_dev, 1009 ) ( A%val(i), i = 1, m*n )
      CASE ( 'SPARSE_BY_ROWS' )
        WRITE( i_dev, 1010 ) ( A%col(i), A%val(i), i=1,A%ptr(m+1)-1 )
        WRITE( i_dev, 1011 ) A%ptr( 1:m+1 )
      CASE ( 'SPARSE_BY_COLUMNS' )
        WRITE( i_dev, 1012 ) ( A%row(i), A%val(i), i = 1, A%ptr(n+1)-1 )
        WRITE( i_dev, 1011 ) A%ptr( 1: n+1 )
      CASE ( 'COORDINATE' )
        WRITE( i_dev, 1013 ) ( A%row(i), A%col(i), A%val(i), i = 1,A%ne )
      CASE ( 'DIAGONAL' )
        WRITE( i_dev, 1009 ) ( A%val(i), i = 1,m )
      CASE DEFAULT
        WRITE( e_dev, 1000 )
        GOTO 999
      END SELECT
    END IF
  END IF

! *****************************
!  Compute row-wise two norms *
! *****************************

  SELECT CASE ( SMT_get( A%type ) )
  CASE ( 'DENSE' )
    IF ( symm ) THEN
      l = 0
      DO i = 1, m
        DO j = 1, i
          l = l + 1 ; val = A%val( l ) ** 2
          row_norms( i ) = row_norms( i ) + val
          IF ( i /= j ) row_norms( j ) = row_norms( j ) + val
        END DO
      END DO
    ELSE
      l = 1
      DO i = 1, m
        l2 = l + n - 1
        row_norms( i ) = DOT_PRODUCT( A%val( l : l2 ), A%val( l : l2 ) )
        l = l + n
      END DO
    END IF
  CASE ( 'SPARSE_BY_ROWS' )
    IF ( symm ) THEN
      row_norms = zero
      DO i = 1, m
        l1 = A%ptr( i ) ; l2 = A%ptr( i + 1 ) - 1
        DO l = l1, l2
          j = A%col( l ) ; val = A%val( l ) ** 2
          row_norms( i ) = row_norms( i ) + val
          IF ( i /= j ) row_norms( j ) = row_norms( j ) + val
        END DO
      END DO
    ELSE
      DO i = 1, m
        l1 = A%ptr( i ) ; l2 = A%ptr( i + 1 ) - 1
        IF ( l2 >= l1 ) THEN
          row_norms( i ) = DOT_PRODUCT( A%val( l1 : l2 ), A%val( l1 : l2 ) )
        ELSE
          row_norms( i )  = zero
        END IF
      END DO
    END IF
    row_norms = SQRT( row_norms )
  CASE ( 'SPARSE_BY_COLUMNS' )
    IF ( symm ) THEN
      row_norms = zero
      DO j = 1, n
        l1 = A%ptr( j ) ; l2 = A%ptr( j + 1 ) - 1
        DO l = l1, l2
          i = A%row( l ) ; val = A%val( l ) ** 2
          row_norms( i ) = row_norms( i ) + val
          IF ( i /= j ) row_norms( j ) = row_norms( j ) + val
        END DO
      END DO
    ELSE
      DO j = 1, n
        l1 = A%ptr( j ) ; l2 = A%ptr( j + 1 ) - 1
        IF ( l2 >= l1 ) row_norms( A%row( l1 : l2 ) ) =                        &
          row_norms( A%row( l1 : l2 ) ) + A%val( l1 : l2 ) ** 2
      END DO
    END IF
  CASE ( 'COORDINATE' )
    row_norms = zero
    IF ( symm ) THEN
      DO l = 1, A%ne
        i = A%row( l ) ; j = A%col( l ) ; val = A%val( l ) ** 2
        row_norms( i ) = row_norms( i ) + val
        IF ( i /= j ) row_norms( j ) = row_norms( j ) + val
      END DO
    ELSE
      DO l = 1, A%ne
        i = A%row( l )
        row_norms( i ) = row_norms( i ) + A%val( l ) ** 2
      END DO
    END IF
    row_norms = SQRT( row_norms )
  CASE ( 'DIAGONAL' )
    row_norms( : n ) = ABS( A%val( : n ) )
  CASE ( 'SCALED_IDENTITY' )
    row_norms( : n ) = ABS( A%val( 1 ) )
  CASE ( 'IDENTITY' )
    row_norms( : n ) = one
  CASE( 'NONE', 'ZERO' )
    row_norms( : n ) = zero
  CASE DEFAULT
    WRITE( e_dev, 1000 )
  END SELECT

999  CONTINUE

  ! possibly print some more.

  IF ( printing >= 1 ) THEN
    WRITE( out, 1014 ) row_norms
    WRITE( out, 1020 )
  END IF

  RETURN

!  Format statements

1000  FORMAT(/,5X,'*** ERROR : mop_row_2_norms : Unrecognized value A%type.')
1001  FORMAT(/,5X,'*** ERROR : mop_row_2_norms : A%m <= 0 and/or A%n <= 0 .')
1002  FORMAT(/,5X,'*** ERROR : mop_row_2_norms : ',                            &
                  'symmetric = .TRUE., but A%m /= A%n.',/)
1005  FORMAT(/,                                                                &
      3X,'*************************************************************',/,    &
!      3X,'*                                                           *',/,   &
      3X,'*                  BEGIN: mop_row_2_norms                   *',/,    &
!      3X,'*                                                           *',/,   &
      3X,'*     GALAHAD computes row-wise norms of a sparse matrix    *',/,    &
!      3X,'*                                                           *',/,   &
      3X,'*************************************************************',/)
1008  FORMAT(/,                                                                &
      5X,'m =', I6, 5X, 'n =', I6, 5X, 'symmetric = ', L1  )
1009  FORMAT(/,5X,'      A%val    ',/, &
               5X,'  -------------',/, (5X, ES17.10 ) )
1010  FORMAT(/,5X,'  A%col             A%val    ',/,  &
               5X,'  -----         -------------',/,  &
              (5X, I7, 7X, ES17.10) )
1011  FORMAT(/,5X,'  A%ptr',/, &
               5X,'  -----',/, (5X, I7 ) )
1012  FORMAT(/,5X,'  A%row             A%val    ',/,  &
               5X,'  -----         -------------',/,  &
              (5X, I7, 7X, ES17.10) )
1013  FORMAT(/,5X,'  A%row         A%col             A%val    ',/,             &
               5X,'  -----         -----         -------------',/,             &
              (5X, I7, 7X, I7, 7X, ES17.10) )
1014  FORMAT(/,5X,'ON EXIT:  row_norms = ', (T28, ES16.9) )
1020  FORMAT(/,                                                               &
      3X,'*************************************************************',/,   &
!     3X,'*                                                           *',/,   &
      3X,'*                   END: mop_row_2_norms                    *',/,   &
!     3X,'*                                                           *',/,   &
!     3X,'*     GALAHAD computes row_wise norms of a sparse matrix    *',/,   &
!     3X,'*                                                           *',/,   &
      3X,'*************************************************************',/)


   END SUBROUTINE mop_row_2_norms

!-*-  B E G I N  m o p _ r o w _ o n e _ n o r m s   S U B R O U T I N E  -*-

   SUBROUTINE mop_row_one_norms( A, row_norms, symmetric, out, error,          &
                                 print_level )
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ...................................................................
!      .                                                                 .
!      .  Returns the vector of row-wise one norms of the matrix A.      .
!      .                                                                 .
!      ...................................................................

!  Arguments:
!  =========

!   A        a scalar variable of derived type SMT_type.
!
!   row_norms  rank 1 array of type real.  The value row_norms(i) gives the
!              one-norm of the i-th row.
!
!   symmetric (optional) is a scalar variable of type logical.  Set
!             symmetric = .TRUE. if the matrix A is symmetric; otherwise
!             set symmetric = .FALSE.  If not present, then the
!             value symmetric = .FALSE. is assumed.
!
!   out      (optional) is a scalar variable of type integer, which
!            holds the stream number for informational messages;  the
!            file is assumed to already have been opened.  Default
!            is out = 6.
!
!   error    (optional) is a scalar variable of type integer, which
!            holds the stream number for error messages; the file
!            is assumed to already have been opened.  Default is error = 6.
!
!   print_level  (optional) is scalar variable of type integer.  It
!                controls the amount of printed information to unit
!                number defined by out.  Values and their meanings are:
!
!                print_level <= 0    Nothing.
!                print_level  = 1    The following is printed: A%m, A%n,
!                                    A%type, A%id, row, col, and val.
!                print_level  = 2    Those from above, in additon to:
!                                    A%ptr, A%val, A%row, and A%col.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


!  Dummy arguments

  TYPE( SMT_type), INTENT( IN  ) :: A
  REAL( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: row_norms
  LOGICAL, INTENT( IN ), OPTIONAL :: symmetric
  INTEGER, INTENT( IN ), OPTIONAL :: error, out, print_level

! Set Parameters

  REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
  REAL ( KIND = wp ), PARAMETER :: one  = 1.0_wp

!  Local variables

  INTEGER :: i, m, n
  INTEGER :: i_dev, e_dev, printing
  LOGICAL :: symm

!**************************************************************

  ! Check for optional arguments.

  if ( present( symmetric ) ) then
     symm = symmetric
  else
     symm = .false.
  end if

  if ( present( error ) ) then
     e_dev = error
  else
     e_dev = 6
  end if

  if ( present( out ) ) then
     i_dev = out
  else
     i_dev = 6
  end if

  if ( present( print_level ) ) then
     printing = print_level
  else
     printing = 0
  end if

  ! Print Header

  IF ( printing >= 1 ) WRITE( i_dev, "(                                        &
 &   3X,'***************************************************************',/,   &
!&   3X,'*                                                             *',/,   &
 &   3X,'*                  BEGIN: mop_row_one_norms                   *',/,   &
!&   3X,'*                                                             *',/,   &
 &   3X,'*   GALAHAD computes row-wise one norms of a sparse matrix    *',/,   &
!&   3X,'*                                                             *',/,   &
 &   3X,'***************************************************************',/)" )

  ! Set some convenient variables.

  m = A%m
  n = A%n

  ! Check for bad dimensions.

  IF ( ( m <= 0 ) .OR. ( n <= 0 ) ) THEN
     WRITE( e_dev, "(/,5X,'*** ERROR : mop_row_one_norms : A%m <= 0',          &
    &  ' and/or A%n <= 0 .')" )
     GOTO 999
  END IF

  IF ( symm ) THEN
     IF ( m /= n ) THEN
        WRITE( e_dev, "(/,5X,'*** ERROR : mop_row_one_norms : ',               &
       &          'symmetric = .TRUE., but A%m /= A%n.',/)" )
        GOTO 999
     END IF
  END IF

! Print information according to variable printing.

  IF ( printing >= 1 ) THEN

     WRITE( i_dev, "(5X, 'A%type = ', 20a)" ) A%type

     IF ( ALLOCATED( A%id ) ) THEN
        WRITE( i_dev, "(5X, 'A%id   = ', 20a)" ) A%id
     ELSE
        WRITE( i_dev, "(5X, 'A%id   =' )")
     END IF

     WRITE( i_dev, "(/, 5X,'m =', I6, 5X, 'n =', I6, 5X, 'symmetric = ',       &
    &                L1  )" ) m, n, symm

     IF ( printing >= 2 ) THEN

        SELECT CASE ( SMT_get( A%type ) )

        CASE ( 'DENSE' )
           WRITE( i_dev, 2010 ) &
                ( A%val(i), i = 1, m*n )
        CASE ( 'SPARSE_BY_ROWS' )
           WRITE( i_dev, "(/,5X,'  A%col             A%val    ',/,             &
          &    5X,'  -----         -------------',/,                           &
          &   (5X, I7, 7X, ES17.10) )" )                                       &
            ( A%col(i), A%val(i), i=1,A%ptr(m+1)-1 )
           WRITE( i_dev, 2020 ) A%ptr( 1:m+1 )
        CASE ( 'SPARSE_BY_COLUMNS' )
           WRITE( i_dev, "(/,5X,'  A%row             A%val',/,                 &
          &     5X,'  -----         -------------',/,                          &
          &    (5X, I7, 7X, ES17.10) )" )                                      &
                ( A%row(i), A%val(i), i = 1, A%ptr(n+1)-1 )
           WRITE( i_dev, 2020 ) A%ptr( 1: n+1 )
        CASE ( 'COORDINATE' )
           WRITE( i_dev, "(/,5X,'  A%row         A%col             A%val',/,   &
          &    5X,'  -----         -----         -------------',/,             &
          &    (5X, I7, 7X, I7, 7X, ES17.10) )" )                              &
             ( A%row(i), A%col(i), A%val(i), i = 1,A%ne )
        CASE ( 'DIAGONAL' )
           WRITE( i_dev, 2010 ) &
                ( A%val(i), i = 1,m )
        CASE DEFAULT
           WRITE( e_dev, 2000 )
           GOTO 999
        END SELECT

     END IF

  END IF

  !******************************
  ! Compute row-wise one norms. !
  !******************************

  SELECT CASE ( SMT_get( A%type ) )

  CASE ( 'DENSE' )
  !***************

     write(*,*) ' WARNING: MOP_ROW_one_NORMS: NOT YET IMPLEMENTED'

  CASE ( 'SPARSE_BY_ROWS' )
  !************************

     write(*,*) ' WARNING: MOP_ROW_one_NORMS: NOT YET IMPLEMENTED'

  CASE ( 'SPARSE_BY_COLUMNS' )
  !***************************

     write(*,*) ' WARNING: MOP_ROW_one_NORMS: NOT YET IMPLEMENTED'

  CASE ( 'COORDINATE' )
  !********************

     row_norms = zero

     if ( symm ) then
        do i = 1, A%ne
           row_norms( A%row(i) ) = row_norms( A%row(i) ) + abs( A%val(i) )
           if ( A%row(i) /= A%col(i) ) then
              row_norms( A%col(i) ) = row_norms( A%col(i) ) + abs( A%val(i) )
           end if
        end do
     else
        do i = 1, A%ne
           row_norms( A%row(i) ) = row_norms( A%row(i) ) + abs( A%val(i) )
        end do
     end if

  CASE ( 'DIAGONAL' )
  !******************

     row_norms( : n ) = ABS( A%val( : n ) )

  CASE ( 'SCALED_IDENTITY' )
  !******************

     row_norms( : n ) = ABS( A%val( 1 ) )

  CASE ( 'IDENTITY' )
  !******************

     row_norms( : n ) = one

   CASE( 'NONE', 'ZERO' )
  !******************

     row_norms( : n ) = zero

  CASE DEFAULT
  !***********

     write( e_dev, 2000 )

  END SELECT

999  CONTINUE

  ! possibly print some more.

  IF ( printing >= 1 ) THEN
     WRITE( out, "(/,5X,'ON EXIT:  row_norms = ', (T28, ES16.9) )" ) row_norms
     WRITE( out, "(                                                            &
    &  3X,'*************************************************************',/,   &
    &  3X,'*                  END: mop_row_one_norms                   *',/,   &
    &  3X,'*************************************************************',/)" )
  END IF

  RETURN

!  Format statements

2000  FORMAT(/,5X,'*** ERROR : mop_row_one_norms : Unrecognized value A%type.')
2010  FORMAT(/,5X,'      A%val    ',/, 5X,'  -------------',/, ( 5X, ES17.10 ) )
2020  FORMAT(/,5X,'  A%ptr',/, 5X,'  -----',/, (5X, I7 ) )


   END SUBROUTINE mop_row_one_norms

!  B E G I N  m o p _ r o w _ i n f i n i t y _ n o r m s   S U B R O U T I N E

   SUBROUTINE mop_row_infinity_norms( A, row_norms, symmetric, out, error,     &
                                      print_level )
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ...................................................................
!      .                                                                 .
!      .  Returns the vector of row-wise infinity norms of the matrix A. .
!      .                                                                 .
!      ...................................................................

!  Arguments:
!  =========

!   A        a scalar variable of derived type SMT_type.
!
!   row_norms  rank 1 array of type real.  The value row_norms(i) gives the
!              infinity-norm of the i-th row.
!
!   symmetric (optional) is a scalar variable of type logical.  Set
!             symmetric = .TRUE. if the matrix A is symmetric; otherwise
!             set symmetric = .FALSE.  If not present, then the
!             value symmetric = .FALSE. is assumed.
!
!   out      (optional) is a scalar variable of type integer, which
!            holds the stream number for informational messages;  the
!            file is assumed to already have been opened.  Default
!            is out = 6.
!
!   error    (optional) is a scalar variable of type integer, which
!            holds the stream number for error messages; the file
!            is assumed to already have been opened.  Default is error = 6.
!
!   print_level  (optional) is scalar variable of type integer.  It
!                controls the amount of printed information to unit
!                number defined by out.  Values and their meanings are:
!
!                print_level <= 0    Nothing.
!                print_level  = 1    The following is printed: A%m, A%n,
!                                    A%type, A%id, row, col, and val.
!                print_level  = 2    Those from above, in additon to:
!                                    A%ptr, A%val, A%row, and A%col.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


!  Dummy arguments

  TYPE( SMT_type), INTENT( IN  ) :: A
  REAL( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: row_norms
  LOGICAL, INTENT( IN ), OPTIONAL :: symmetric
  INTEGER, INTENT( IN ), OPTIONAL :: error, out, print_level

! Set Parameters

  REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
  REAL ( KIND = wp ), PARAMETER :: one  = 1.0_wp

!  Local variables

  INTEGER :: i, j, l, m, n
  INTEGER :: i_dev, e_dev, printing
  LOGICAL :: symm

!**************************************************************

  ! Check for optional arguments.

  IF ( PRESENT( symmetric ) ) THEN
     symm = symmetric
  ELSE
     symm = .false.
  END IF

  IF ( PRESENT( error ) ) THEN
     e_dev = error
  ELSE
     e_dev = 6
  END IF

  IF ( PRESENT( out ) ) THEN
     i_dev = out
  ELSE
     i_dev = 6
  END IF

  IF ( PRESENT( print_level ) ) THEN
     printing = print_level
  ELSE
     printing = 0
  END IF

  ! Print Header

  IF ( printing >= 1 ) WRITE( i_dev, "(                                        &
 &   3X,'*****************************************************************',/, &
!&   3X,'*                                                               *',/, &
 &   3X,'*                 BEGIN: mop_row_infinity_norms                 *',/, &
!&   3X,'*                                                               *',/, &
 &   3X,'*  GALAHAD computes row-wise infinity norms of a sparse matrix  *',/, &
!&   3X,'*                                                               *',/, &
 &   3X,'*****************************************************************',/)")

  ! Set some convenient variables.

  m = A%m
  n = A%n

  ! Check for bad dimensions.

  IF ( ( m <= 0 ) .OR. ( n <= 0 ) ) THEN
     WRITE( e_dev, "(/,5X, '*** ERROR : mop_row_infinity_norms : A%m <= 0',    &
    &  ' and/or A%n <= 0')" )
     GO TO 999
  END IF

  IF ( symm ) THEN
     IF ( m /= n ) THEN
        WRITE( e_dev, "(/, 5X,'*** ERROR : mop_row_infinity_norms : ',         &
       &          'symmetric = .TRUE., but A%m /= A%n', / )" )
        GO TO 999
     END IF
  END IF

! Print information according to variable printing.

  IF ( printing >= 1 ) THEN
     WRITE( i_dev, "(5X, 'A%type = ', 20a)" ) A%type

     IF ( ALLOCATED( A%id ) ) THEN
        WRITE( i_dev, "(5X, 'A%id   = ', 20a)" ) A%id
     ELSE
        WRITE( i_dev, "(5X, 'A%id   =' )")
     END IF

     WRITE( i_dev, "(/, 5X,'m =', I6, 5X, 'n =', I6, 5X, 'symmetric = ',       &
    &                L1  )" ) m, n, symm

     IF ( printing >= 2 ) THEN
        SELECT CASE ( SMT_get( A%type ) )

        CASE ( 'DENSE' )
           WRITE( i_dev, 2010 ) &
                ( A%val(i), i = 1, m*n )
        CASE ( 'SPARSE_BY_ROWS' )
           WRITE( i_dev, "(/,5X,'  A%col             A%val    ',/,             &
          &    5X,'  -----         -------------',/,                           &
          &   (5X, I7, 7X, ES17.10) )" )                                       &
            ( A%col(i), A%val(i), i=1,A%ptr(m+1)-1 )
           WRITE( i_dev, 2020 ) A%ptr( 1:m+1 )
        CASE ( 'SPARSE_BY_COLUMNS' )
           WRITE( i_dev, "(/,5X,'  A%row             A%val',/,                 &
          &     5X,'  -----         -------------',/,                          &
          &    (5X, I7, 7X, ES17.10) )" )                                      &
                ( A%row(i), A%val(i), i = 1, A%ptr(n+1)-1 )
           WRITE( i_dev, 2020 ) A%ptr( 1: n+1 )
        CASE ( 'COORDINATE' )
           WRITE( i_dev, "(/,5X,'  A%row         A%col             A%val',/,   &
          &    5X,'  -----         -----         -------------',/,             &
          &    (5X, I7, 7X, I7, 7X, ES17.10) )" )                              &
             ( A%row(i), A%col(i), A%val(i), i = 1,A%ne )
        CASE ( 'DIAGONAL' )
           WRITE( i_dev, 2010 ) &
                ( A%val(i), i = 1,m )
        CASE DEFAULT
           WRITE( e_dev, 2000 )
           GOTO 999
        END SELECT

     END IF

  END IF

  !******************************
  ! Compute row-wise infinity norms. !
  !******************************

  SELECT CASE ( SMT_get( A%type ) )

  CASE ( 'DENSE' )
  !***************

     WRITE(*,*) ' WARNING: MOP_ROW_infinity_NORMS: NOT YET IMPLEMENTED'

  CASE ( 'SPARSE_BY_ROWS' )
  !************************

     WRITE(*,*) ' WARNING: MOP_ROW_infinity_NORMS: NOT YET IMPLEMENTED'

  CASE ( 'SPARSE_BY_COLUMNS' )
  !***************************

     WRITE(*,*) ' WARNING: MOP_ROW_infinity_NORMS: NOT YET IMPLEMENTED'

  CASE ( 'COORDINATE' )
  !********************

     row_norms = zero

     IF ( symm ) THEN
        DO l = 1, A%ne
         i = A%row( l ) ; j = A%col( l )
         row_norms( i ) = MAX( row_norms( i ), ABS( A%val( l ) ) )
         IF ( i /= j ) row_norms( j ) = MAX( row_norms( j ), ABS( A%val( l ) ) )
        END DO
     ELSE
        DO l = 1, A%ne
         i = A%row( l )
         row_norms( i ) = MAX( row_norms( i ), ABS( A%val( l ) ) )
        END DO
     END IF

  CASE ( 'DIAGONAL' )
  !******************

     row_norms( : n ) = ABS( A%val( : n ) )

  CASE ( 'SCALED_IDENTITY' )
  !******************

     row_norms( : n ) = ABS( A%val( 1 ) )

  CASE ( 'IDENTITY' )
  !******************

     row_norms( : n ) = one

   CASE( 'NONE', 'ZERO' )
  !******************

     row_norms( : n ) = zero

  CASE DEFAULT
  !***********

     WRITE( e_dev, 2000 )

  END SELECT

999  CONTINUE

  ! possibly print some more.

  IF ( printing >= 1 ) THEN
     WRITE( out, "(/,5X,'ON EXIT:  row_norms = ', (T28, ES16.9) )" ) row_norms
     WRITE( out, "(                                                            &
    &  3X,'***************************************************************',/, &
    &  3X,'*                END: mop_row_infinity_norms                  *',/, &
    &  3X,'***************************************************************',/)")
  END IF

  RETURN

!  Format statements

2000  FORMAT(/,5X,'*** ERROR : mop_row_one_norms : Unrecognized value A%type.')
2010  FORMAT(/,5X,'      A%val    ',/, &
               5X,'  -------------',/, (5X, ES17.10 ) )
2020  FORMAT(/,5X,'  A%ptr',/, 5X,'  -----',/, (5X, I7 ) )


   END SUBROUTINE mop_row_infinity_norms

!-*-  B E G I N  m o p _ c o l u m n _ 2 _ n o r m s   S U B R O U T I N E  -*-

   SUBROUTINE mop_column_2_norms( A, column_norms, W, symmetric, out, error,   &
                                  print_level )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!   .......................................................................
!   .                                                                     .
!   .  Returns the vector of column-wise two norms of the matrix W^1/2 A. .
!   .                                                                     .
!   .......................................................................

!  Arguments:
!  =========

!   A        a scalar variable of derived type SMT_type.
!
!   column_norms  rank 1 array of type real. The value column_norms(i)
!              gives the 2-norm of the i-th column.
!
!   W (optional) rank 1 array of type real. The positive value W(i) specifies
!              the ith entry of the diagonal matrix W.

!   symmetric (optional) is a scalar variable of type logical.  Set
!             symmetric = .TRUE. if the matrix A is symmetric; otherwise
!             set symmetric = .FALSE.  If not present, then the
!             value symmetric = .FALSE. is assumed.
!
!   out      (optional) is a scalar variable of type integer, which
!            holds the stream number for informational messages;  the
!            file is assumed to already have been opened.  Default
!            is out = 6.
!
!   error    (optional) is a scalar variable of type integer, which
!            holds the stream number for error messages; the file
!            is assumed to already have been opened.  Default is error = 6.
!
!   print_level  (optional) is scalar variable of type integer.  It
!                controls the amount of printed information to unit
!                number defined by out.  Values and their meanings are:
!
!                print_level <= 0    Nothing.
!                print_level  = 1    The following is printed: A%m, A%n,
!                                    A%type, A%id, row, col, and val.
!                print_level  = 2    Those from above, in additon to:
!                                    A%ptr, A%val, A%row, and A%col.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

  TYPE( SMT_type), INTENT( IN  ) :: A
  REAL( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: column_norms
  REAL( KIND = wp ), DIMENSION( : ), INTENT( IN ), OPTIONAL :: W
  LOGICAL, INTENT( IN ), OPTIONAL :: symmetric
  INTEGER, INTENT( IN ), OPTIONAL :: error, out, print_level

! Set Parameters

  REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
  REAL ( KIND = wp ), PARAMETER :: one  = 1.0_wp

!  Local variables

  INTEGER :: i, j, l, l1, l2, m, n
  INTEGER :: i_dev, e_dev, printing
  REAL ( KIND = wp ) :: val
  LOGICAL :: symm, w_eq_identity

!  Check for optional arguments

  w_eq_identity = .NOT. PRESENT( W )
  IF ( PRESENT( symmetric ) ) THEN
    symm = symmetric
  ELSE
    symm = .false.
  END IF
  IF ( PRESENT( error ) ) THEN
    e_dev = error
  ELSE
    e_dev = 6
  END if
  IF ( PRESENT( out ) ) THEN
    i_dev = out
  ELSE
    i_dev = 6
  END IF
    IF ( PRESENT( print_level ) ) THEN
    printing = print_level
  ELSE
    printing = 0
  END IF

!   Print Header

  IF ( printing >= 1 ) WRITE( i_dev, 1005 )

!   Set some convenient variables

  m = A%m ; n = A%n

!   Check for bad dimensions

  IF ( ( m <= 0 ) .OR. ( n <= 0 ) ) THEN
    WRITE( e_dev, 1001 )
    GOTO 999
  END IF

  IF ( symm ) THEN
    IF ( m /= n ) THEN
      WRITE( e_dev, 1002 )
      GOTO 999
    END IF
  END IF

!  Print information according to variable printing

  IF ( printing >= 1 ) THEN
    WRITE( i_dev, "(5X, 'A%type = ', 20a)" ) A%type
    IF ( ALLOCATED( A%id ) ) THEN
       WRITE( i_dev, "(5X, 'A%id   = ', 20a)" ) A%id
    ELSE
       WRITE( i_dev, "(5X, 'A%id   =' )")
    END IF

    WRITE( i_dev, 1008 ) m, n, symm
    IF ( printing >= 2 ) THEN
      SELECT CASE ( SMT_get( A%type ) )
      CASE ( 'DENSE' )
        WRITE( i_dev, 1009 ) ( A%val(i), i = 1, m*n )
      CASE ( 'SPARSE_BY_ROWS' )
        WRITE( i_dev, 1010 ) ( A%col(i), A%val(i), i=1,A%ptr(m+1)-1 )
        WRITE( i_dev, 1011 ) A%ptr( 1:m+1 )
      CASE ( 'SPARSE_BY_COLUMNS' )
        WRITE( i_dev, 1012 ) ( A%row(i), A%val(i), i = 1, A%ptr(n+1)-1 )
        WRITE( i_dev, 1011 ) A%ptr( 1: n+1 )
      CASE ( 'COORDINATE' )
        WRITE( i_dev, 1013 ) ( A%row(i), A%col(i), A%val(i), i = 1,A%ne )
      CASE ( 'DIAGONAL' )
        WRITE( i_dev, 1009 ) ( A%val(i), i = 1,m )
      CASE DEFAULT
        WRITE( e_dev, 1000 )
        GOTO 999
      END SELECT
    END IF
  END IF

! ********************************
!  Compute column-wise two norms *
! ********************************

  SELECT CASE ( SMT_get( A%type ) )
  CASE ( 'DENSE' )
    IF ( symm ) THEN
      IF ( w_eq_identity ) THEN
        l = 0
        DO i = 1, m
          DO j = 1, i
            l = l + 1 ; val = A%val( l ) ** 2
            column_norms( j ) = column_norms( j ) + val
            IF ( i /= j ) column_norms( i ) = column_norms( i ) + val
          END DO
        END DO
      ELSE
        l = 0
        DO i = 1, m
          DO j = 1, i
            l = l + 1
            column_norms( j ) = column_norms( j ) + W( i ) * A%val( l ) ** 2
            IF ( i /= j )                                                      &
              column_norms( i ) = column_norms( i ) + W( j ) * A%val( l ) ** 2
          END DO
        END DO
      END IF
    ELSE
      column_norms = zero
      IF ( w_eq_identity ) THEN
        l = 1
        DO i = 1, m
          l2 = l + n - 1
          column_norms( : n ) = column_norms( : n ) + A%val( l : l2 ) ** 2
          l = l + n
        END DO
      ELSE
        l = 1
        DO i = 1, m
          DO j = 1, n
            l = l + 1
            column_norms( j ) = column_norms( j ) + W( i ) * A%val( l ) ** 2
          END DO
        END DO
      END IF
    END IF
  CASE ( 'SPARSE_BY_ROWS' )
    IF ( symm ) THEN
      column_norms = zero
      IF ( w_eq_identity ) THEN
        DO i = 1, m
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            j = A%col( l ) ; val = A%val( l ) ** 2
            column_norms( j ) = column_norms( j ) + val
            IF ( i /= j ) column_norms( i ) = column_norms( i ) + val
          END DO
        END DO
      ELSE
        DO i = 1, m
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            j = A%col( l )
            column_norms( j ) = column_norms( j ) + W( i ) * A%val( l ) ** 2
            IF ( i /= j )                                                      &
              column_norms( i ) = column_norms( i ) + W( j ) * A%val( l ) ** 2
          END DO
        END DO
      END IF
    ELSE
      IF ( w_eq_identity ) THEN
        DO i = 1, m
          l1 = A%ptr( i ) ; l2 = A%ptr( i + 1 ) - 1
          IF ( l2 >= l1 ) column_norms( A%col( l1 : l2 ) ) =                   &
              column_norms( A%col( l1 : l2 ) ) + A%val( l1 : l2 ) ** 2
        END DO
      ELSE
        DO i = 1, m
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            j = A%col( l )
            column_norms( j ) = column_norms( j ) + W( i ) * A%val( l ) ** 2
          END DO
        END DO
      END IF
    END IF
  CASE ( 'SPARSE_BY_COLUMNS' )
    IF ( symm ) THEN
      column_norms = zero
      IF ( w_eq_identity ) THEN
        DO j = 1, n
          DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
            i = A%row( l ) ; val = A%val( l ) ** 2
            column_norms( j ) = column_norms( j ) + val
            IF ( i /= j ) column_norms( i ) = column_norms( i ) + val
          END DO
        END DO
      ELSE
        DO j = 1, n
          DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
            i = A%row( l )
            column_norms( j ) = column_norms( j ) + W( i ) * A%val( l ) ** 2
            IF ( i /= j )                                                      &
              column_norms( i ) = column_norms( i ) + W( j ) * A%val( l ) ** 2
          END DO
        END DO
      END IF
    ELSE
      IF ( w_eq_identity ) THEN
        DO j = 1, n
          l1 = A%ptr( j ) ; l2 = A%ptr( j + 1 ) - 1
          IF ( l2 >= l1 ) THEN
            column_norms( j ) = DOT_PRODUCT( A%val( l1 : l2 ), A%val( l1 : l2 ))
          ELSE
            column_norms( j )  = zero
          END IF
        END DO
      ELSE
        DO j = 1, n
          DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
            i = A%row( l )
            column_norms( j ) = column_norms( j ) + W( i ) * A%val( l ) ** 2
          END DO
        END DO
      END IF
    END IF
    column_norms = SQRT( column_norms )
  CASE ( 'COORDINATE' )
    column_norms = zero
    IF ( symm ) THEN
      IF ( w_eq_identity ) THEN
        DO l = 1, A%ne
          i = A%row( l ) ; j = A%col( l ) ; val = A%val( l ) ** 2
          column_norms( j ) = column_norms( j ) + val
          IF ( i /= j ) column_norms( i ) = column_norms( i ) + val
        END DO
      ELSE
        DO l = 1, A%ne
          i = A%row( l ) ; j = A%col( l )
          column_norms( j ) = column_norms( j ) + W( i ) * A%val( l ) ** 2
          IF ( i /= j )                                                        &
            column_norms( i ) = column_norms( i ) + W( j ) * A%val( l ) ** 2
        END DO
      END IF
    ELSE
      IF ( w_eq_identity ) THEN
        DO l = 1, A%ne
          j = A%col( l )
          column_norms( j ) = column_norms( j ) + A%val( l ) ** 2
        END DO
      ELSE
        DO l = 1, A%ne
          i = A%row( l ) ; j = A%col( l )
          column_norms( j ) = column_norms( j ) + W( i ) * A%val( l ) ** 2
        END DO
      END IF
    END IF
    column_norms = SQRT( column_norms )
  CASE ( 'DIAGONAL' )
    IF ( w_eq_identity ) THEN
      column_norms( : n ) = ABS( A%val( : n ) )
    ELSE
      column_norms( : n ) = ABS( SQRT( W( : n ) ) * A%val( : n ) )
    END IF
  CASE ( 'SCALED_IDENTITY' )
    IF ( w_eq_identity ) THEN
      column_norms( : n ) = ABS( SQRT( W( : n ) ) * A%val( 1 ) )
    ELSE
    END IF
  CASE ( 'IDENTITY' )
    IF ( w_eq_identity ) THEN
      column_norms( : n ) = one
    ELSE
      column_norms( : n ) = SQRT( W( : n ) )
    END IF
  CASE( 'NONE', 'ZERO' )
    column_norms( : n ) = zero
  CASE DEFAULT
    WRITE( e_dev, 1000 )
  END SELECT

999  CONTINUE

  ! possibly print some more.

  IF ( printing >= 1 ) THEN
    WRITE( out, 1014 ) column_norms
    WRITE( out, 1020 )
  END IF

  RETURN

!  Format statements

1000  FORMAT(/,5X,'*** ERROR : mop_column_2_norms : Unrecognized value A%type.')
1001  FORMAT(/,5X,'*** ERROR : mop_column_2_norms : A%m <= 0 and/or A%n <= 0 .')
1002  FORMAT(/,5X,'*** ERROR : mop_column_2_norms : ',                         &
                  'symmetric = .TRUE., but A%m /= A%n.',/)
1005  FORMAT(/,                                                                &
      3X,'*************************************************************',/,    &
!      3X,'*                                                           *',/,   &
      3X,'*                  BEGIN: mop_column_2_norms                   *',/, &
!      3X,'*                                                           *',/,   &
      3X,'*     GALAHAD computes column-wise norms of a sparse matrix    *',/, &
!      3X,'*                                                           *',/,   &
      3X,'*************************************************************',/)
1008  FORMAT(/,                                                                &
      5X,'m =', I6, 5X, 'n =', I6, 5X, 'symmetric = ', L1  )
1009  FORMAT(/,5X,'      A%val    ',/, &
               5X,'  -------------',/, (5X, ES17.10 ) )
1010  FORMAT(/,5X,'  A%col             A%val    ',/,  &
               5X,'  -----         -------------',/,  &
              (5X, I7, 7X, ES17.10) )
1011  FORMAT(/,5X,'  A%ptr',/, &
               5X,'  -----',/, (5X, I7 ) )
1012  FORMAT(/,5X,'  A%row             A%val    ',/,  &
               5X,'  -----         -------------',/,  &
              (5X, I7, 7X, ES17.10) )
1013  FORMAT(/,5X,'  A%row         A%col             A%val    ',/,             &
               5X,'  -----         -----         -------------',/,             &
              (5X, I7, 7X, I7, 7X, ES17.10) )
1014  FORMAT(/,5X,'ON EXIT:  column_norms = ', (T28, ES16.9) )
1020  FORMAT(/,                                                                &
      3X,'*************************************************************',/,    &
!     3X,'*                                                           *',/,    &
      3X,'*                   END: mop_column_2_norms                    *',/, &
!     3X,'*                                                           *',/,    &
!     3X,'*     GALAHAD computes column_wise norms of a sparse matrix    *',/, &
!     3X,'*                                                           *',/,    &
      3X,'*************************************************************',/)


   END SUBROUTINE mop_column_2_norms

 END MODULE GALAHAD_MOP_double

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*                             *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*     END MOP  M O D U L E    *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*                             *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
