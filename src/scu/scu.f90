! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ S C U   M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released pre GALAHAD Version 1.0. March 25th 1999
!   update released with GALAHAD Version 2.0. March 26th 2005

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_SCU_double

!      -------------------------------------------
!      |                                         |
!      | Solve the augmented system of equations |
!      |                                         |
!      |    / A  B \ / x1 \ _ / rhs1 \           |
!      |    \ C  D / \ x2 / ~ \ rhs2 /           |
!      |                                         |
!      | using the Schur complement method       |
!      |                                         |
!      -------------------------------------------

    USE GALAHAD_BLAS_interface, ONLY : ROT, ROTG

    IMPLICIT NONE

    PRIVATE
    PUBLIC :: SCU_restart_m_eq_0, SCU_factorize, SCU_solve,                    &
              SCU_append, SCU_delete, SCU_increase_diagonal, SCU_terminate

!--------------------
!   P r e c i s i o n
!--------------------

    INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

    REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
    REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
    REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
    REAL ( KIND = wp ), PARAMETER :: r_pos = 0.01_wp

!----------------------------
!   D e r i v e d   T y p e s 
!----------------------------

    TYPE, PUBLIC :: SCU_matrix_type
      INTEGER :: n, m, m_max
      INTEGER :: class = 0
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: BD_row
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: BD_col_start
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: CD_col
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: CD_row_start
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: BD_val
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: CD_val
    END TYPE

    TYPE, PUBLIC :: SCU_info_type
      INTEGER :: alloc_status
      INTEGER, DIMENSION( 3 ) :: inertia
    END TYPE

    TYPE, PUBLIC :: SCU_data_type
      PRIVATE
      INTEGER :: m, m_max, jumpto, jcol, newdia, sign_determinant
      INTEGER :: class = 3
      LOGICAL :: got_factors
      REAL ( KIND = wp ) :: dianew
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: Q
    END TYPE

  CONTAINS

!-*-*-*-  S C U _ r e s t a r t _ m _ e q _ 0    S U B R O U T I N E   -*-*-*-*-

    SUBROUTINE SCU_restart_m_eq_0( data, info )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Form and factorize the Schur complement
!
!      S = D - C * A(inverse) * B
!
!  of the matrix A in the symmetric or unsymmetric block matrix
!
!     / A  B \
!     \ C  D /
!
!  Nick Gould, Fortran 77 version August 4th 1988
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( SCU_info_type ), INTENT( INOUT ) :: info
      TYPE ( SCU_data_type ), INTENT( INOUT ) :: data

!  Re-initialize data values

      data%m = 0
      data%got_factors = .TRUE.
      data%sign_determinant = 1
      info%inertia = (/ 0, 0, 0 /)
      RETURN

    END SUBROUTINE SCU_restart_m_eq_0

!-*-*-*-*-*-  S C U _ f a c t o r i z e    S U B R O U T I N E   -*-*-*-*-*-*-

    SUBROUTINE SCU_factorize( matrix, data, VECTOR, status, info )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Form and factorize the Schur complement
!
!      S = D - C * A(inverse) * B
!
!  of the matrix A in the symmetric or unsymmetric block matrix
!
!     / A  B \
!     \ C  D /
!
!  Nick Gould, Fortran 77 version August 4th 1988
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( SCU_matrix_type ), INTENT( INOUT ) :: matrix
      TYPE ( SCU_info_type ), INTENT( INOUT ) :: info
      TYPE ( SCU_data_type ), INTENT( INOUT ) :: data
      INTEGER, INTENT( INOUT ) :: status
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION ( matrix%n ) :: VECTOR

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i, irow, j, k, newclr, newdia, kirn, kjcn, mnew, msofar,      &
                 sign_determinant
      REAL ( KIND = wp ) :: scalar

!-----------------------------------------------
!   E x t e r n a l   F u n c t i o n s
!-----------------------------------------------

      IF ( status <= 0 ) THEN ; status = - 2 ; RETURN ; END IF
      IF ( status == 2 ) GO TO 30

      data%got_factors = .FALSE.
      IF ( matrix%m_max < 0 .OR.  matrix%class < 1 .OR.  matrix%class > 4 )    &
        THEN ; status = - 1 ; RETURN ; ELSE ; status = 0 ; END IF

!  Ensure that there is sufficient space

      IF ( matrix%m < 0 .OR. matrix%m > matrix%m_max .OR. matrix%n < 0 ) THEN
        status = - 1 ; RETURN ; END IF

!  Initialize data values

      data%m = 0
      data%class = matrix%class
      data%m_max = matrix%m_max
      data%sign_determinant = 1
      info%inertia = (/ 0, 0, 0 /)

!  Allocate the data arrays Q, R and W

      IF ( matrix%class <= 2 ) THEN
        IF ( ALLOCATED( data%Q ) ) THEN
          DEALLOCATE( data%Q, STAT = i )
          info%alloc_status = i
          IF ( info%alloc_status /= 0 ) THEN ; status = - 12 ; RETURN ; END IF
        END IF
        ALLOCATE( data%Q( matrix%m_max, matrix%m_max ), STAT = i )
        info%alloc_status = i
        IF ( info%alloc_status /= 0 ) THEN ; status = - 12 ; RETURN ; END IF
      END IF

      IF ( ALLOCATED( data%R ) ) THEN
        DEALLOCATE( data%R, STAT = i )
        info%alloc_status = i
        IF ( info%alloc_status /= 0 ) THEN ; status = - 12 ; RETURN ; END IF
      END IF
      ALLOCATE( data%R( matrix%m_max * ( matrix%m_max + 1 ) / 2 ), STAT = i )
      info%alloc_status = i
      IF ( info%alloc_status /= 0 ) THEN
        IF ( matrix%class <= 2 ) DEALLOCATE( data%Q, STAT = status )
        status = - 12
        RETURN
      END IF

      IF ( ALLOCATED( data%W ) ) THEN
        DEALLOCATE( data%W, STAT = i )
        info%alloc_status = i
        IF ( info%alloc_status /= 0 ) THEN ; status = - 12 ; RETURN ; END IF
      END IF
      ALLOCATE( data%W( matrix%m_max ), STAT = i )
      info%alloc_status = i
      IF ( info%alloc_status /= 0 ) THEN
        IF ( matrix%class <= 2 ) DEALLOCATE( data%Q, STAT = status )
        IF ( matrix%class <= 2 ) DEALLOCATE( data%R, STAT = status )
        status = - 12
      END IF

!  Ensure that array arguments are present when required

      IF ( .NOT. ( ALLOCATED( matrix%BD_val ) .AND.                            &
                   ALLOCATED( matrix%BD_row ) .AND.                            &
                   ALLOCATED( matrix%BD_col_start ) ) ) THEN
        status = - 4 ; RETURN ; END IF

      IF ( data%class == 1 ) THEN
        IF ( .NOT. ( ALLOCATED( matrix%CD_val ) .AND.                          &
                     ALLOCATED( matrix%CD_row_start ) .AND.                    &
                     ALLOCATED( matrix%CD_col ) ) ) THEN 
          status = - 5 ; RETURN ; END IF
      END IF

!  Ensure that the matrices are large enough

      IF ( SIZE( matrix%BD_col_start ) < matrix%m + 1 ) THEN
         status = - 6 ; RETURN
      ELSE IF ( MIN( SIZE( matrix%BD_val ), SIZE( matrix%BD_row ) ) <          &
                   matrix%BD_col_start( matrix%m + 1 ) - 1 ) THEN
         status = - 6 ; RETURN
      END IF

!  Ensure that same is true for optional arrays

      IF ( data%class == 1 ) THEN
        IF ( SIZE( matrix%CD_row_start ) < matrix%m + 1 ) THEN
           status = - 7 ; RETURN
        ELSE IF ( MIN( SIZE( matrix%CD_val ), SIZE( matrix%CD_col ) ) <        &
               matrix%CD_row_start( matrix%m + 1 ) - 1 ) THEN
           status = - 7 ; RETURN
        END IF
      END IF

      data%sign_determinant = 1
      info%inertia = (/ 0, 0, 0 /)

      IF ( matrix%m == 0 ) GO TO 100

!  Remove subdiagonal elements from the data structures associated
!  with B and the upper triangular part of D

      kirn = 1
      DO j = 1, matrix%m
        matrix%BD_col_start( j ) = kirn
        newdia = matrix%n + j
        DO k = matrix%BD_col_start( j ), matrix%BD_col_start( j + 1 ) - 1
          i = matrix%BD_row( k )
          IF ( i <= newdia ) THEN
            matrix%BD_row( kirn ) = i
            matrix%BD_val( kirn ) = matrix%BD_val( k )
            kirn = kirn + 1
          END IF
        END DO
      END DO
      matrix%BD_col_start( matrix%m + 1 ) = kirn

!  Remove diagonal and superdiagonal elements from the data
!  structures associated with C and the lower triangular part of D

      IF ( data%class == 1 ) THEN
        kjcn = 1
        DO irow = 1, matrix%m
          matrix%CD_row_start( irow ) = kjcn
          newdia = matrix%n + irow
          DO k = matrix%CD_row_start( irow ), matrix%CD_row_start( irow + 1) - 1
            j = matrix%CD_col( k )
            IF ( j < newdia ) THEN
              matrix%CD_col( kjcn ) = j
              matrix%CD_val( kjcn ) = matrix%CD_val( k )
              kjcn = kjcn + 1
            END IF
          END DO
        END DO
        matrix%CD_row_start( matrix%m + 1 ) = kjcn
      END IF

!  Form the Schur complement one column at a time in the array Q

      data%jcol = 1
 20   CONTINUE

!  Find the icol-th column of B

      VECTOR( : matrix%n ) = zero

!  S is symmetric and definite

      IF ( data%class > 2 ) THEN
        newclr = data%jcol * ( data%jcol - 1 ) / 2
        data%R( newclr + 1 : newclr + data%jcol ) = zero

!  S is symmetric and positive definite

        IF ( data%class == 3 ) THEN
          DO k = matrix%BD_col_start( data%jcol ),                             &
                 matrix%BD_col_start( data%jcol + 1 ) - 1
            j = matrix%BD_row( k )
            IF ( j <= matrix%n ) THEN
              VECTOR( j ) = matrix%BD_val( k )
            ELSE
              irow = j - matrix%n
              IF ( irow <= data%jcol )                                         &
                data%R( newclr + irow ) = matrix%BD_val( k )
            END IF
          END DO

!  S is symmetric and negative definite
        
        ELSE
          DO k = matrix%BD_col_start( data%jcol ),                             &
                 matrix%BD_col_start( data%jcol + 1 ) - 1
            j = matrix%BD_row( k )
            IF ( j <= matrix%n ) THEN
              VECTOR( j ) = - matrix%BD_val( k )
            ELSE
              irow = j - matrix%n
              IF ( irow <= data%jcol )                                         &
                data%R( newclr + irow ) = - matrix%BD_val( k )
            END IF
          END DO
        END IF
      ELSE
        IF ( data%class == 1 ) THEN
          data%Q( : matrix%m, data%jcol ) = zero
        ELSE
          data%Q( : data%jcol, data%jcol ) = zero
        END IF

!  S is (possibly) indefinite

        DO k = matrix%BD_col_start( data%jcol ),                               &
               matrix%BD_col_start( data%jcol + 1 ) - 1
          j = matrix%BD_row( k )
          IF ( j <= matrix%n ) THEN
            VECTOR( j ) = matrix%BD_val( k )
          ELSE
            irow = j - matrix%n
            IF ( irow <= data%jcol )                                           &
              data%Q( irow, data%jcol ) = matrix%BD_val( k )
          END IF
        END DO
      END IF

!  Return to obtain A(inverse) * new column

      IF ( matrix%n > 0 ) THEN ; status = 2 ; RETURN ; END IF

 30   CONTINUE

!  Obtain the new column of Q. S is indefinite

      IF ( data%class <= 2 ) THEN

!  Obtain the new column of Q. S is unsymmetric

        IF ( data%class == 1 ) THEN
          DO irow = 1, matrix%m
            scalar = data%Q( irow, data%jcol )
            IF ( matrix%n > 0 ) THEN
              DO k = matrix%CD_row_start( irow ),                              &
                     matrix%CD_row_start( irow + 1 ) - 1
                j = matrix%CD_col( k )
                IF ( j - matrix%n == data%jcol )                               &
                  scalar = scalar + matrix%CD_val( k )
                IF ( j <= matrix%n )                                           &
                  scalar = scalar - matrix%CD_val( k ) * VECTOR( j )
              END DO
            END IF
            data%Q( irow, data%jcol ) = scalar
          END DO

!  S is symmetric but indefinite

        ELSE
          DO irow = 1, data%jcol
            scalar = data%Q( irow, data%jcol )
            IF ( matrix%n > 0 ) THEN
              DO k = matrix%BD_col_start( irow ),                              &
                     matrix%BD_col_start( irow + 1 ) - 1
                j = matrix%BD_row( k )
                IF ( j <= matrix%n )                                           &
                  scalar = scalar - matrix%BD_val( k ) * VECTOR( j )
              END DO
            END IF
            data%Q( irow, data%jcol ) = scalar
            data%Q( data%jcol, irow ) = scalar
          END DO
        END IF

!  S is symmetric, definite

      ELSE
        IF ( matrix%n > 0 ) THEN
          newclr = data%jcol * ( data%jcol - 1 ) / 2
          DO irow = 1, data%jcol
            scalar = data%R( newclr + irow )
            DO k = matrix%BD_col_start( irow ),                                &
                   matrix%BD_col_start( irow + 1 ) - 1
              j = matrix%BD_row( k )
              IF ( j <= matrix%n )                                             &
                scalar = scalar - matrix%BD_val( k ) * VECTOR( j )
            END DO
            data%R( newclr + irow ) = scalar
          END DO
        END IF
      END IF
      data%jcol = data%jcol + 1
      IF ( data%jcol <= matrix%m ) GO TO 20
      IF ( data%class <= 2 ) THEN

!  Find the QR factorization of the first msofar rows and columns
!  of the Schur complement

        DO mnew = 1, matrix%m
          msofar = mnew - 1
          newclr = msofar * mnew / 2

!  Initialize the new diagonal of Q

          data%W( mnew ) = data%Q( mnew, mnew )
          data%Q( mnew, mnew ) = one

          IF ( msofar > 0 ) THEN

!  Store the newly introduced row in W.

!           CALL dcopy( msofar, data%Q( mnew, 1 ), lq, data%W, 1 )
            data%W( : msofar ) = data%Q( mnew, 1 : msofar )

!  Multiply the newly introduced column by the Q(transpose) obtained
!  so far and store as the mnew-th column of R

            data%R( newclr + 1 : newclr + msofar ) = zero
            DO irow = 1, msofar
              scalar = data%Q( irow, mnew )
              data%R( newclr + 1 : newclr + msofar ) =                         &
                data%R( newclr + 1 : newclr + msofar )                         &
                  + scalar * data%Q( irow, 1 : msofar )

!  Initialize the new last row and column of Q

              data%Q( irow, mnew ) = zero
              data%Q( mnew, irow ) = zero
            END DO
          END IF

!  Reduce the new row to zero by applying plane-rotation matrices

          CALL SCU_triangular( msofar, 1, data%R, data%W, status,              &
                               Q = data%Q )
          IF ( status < 0 ) THEN
            RETURN
          END IF

!  Determine the inertia of S

          sign_determinant = SCU_sign_determinant( mnew, data%R )
          IF ( sign_determinant == data%sign_determinant ) THEN
            info%inertia( 1 ) = info%inertia( 1 ) + 1
          ELSE
            info%inertia( 2 ) = info%inertia( 2 ) + 1
            data%sign_determinant = sign_determinant
          END IF

        END DO

!  S is allegedly positive definite. Find the R(transpose) R factorization 
!  of the first msofar rows and columns of the Schur complement

      ELSE
        scalar = data%R( 1 )
        IF ( scalar <= zero ) THEN
          IF ( data%class == 3 ) THEN
             status = - 10
          ELSE
             status = - 11
          END IF
          RETURN 
        END IF
        IF ( data%class == 3 ) THEN
          info%inertia( 1 ) = info%inertia( 1 ) + 1
        ELSE
          info%inertia( 2 ) = info%inertia( 2 ) + 1
        END IF
        data%R( 1 ) = sqrt( scalar )
        newdia = 1
        DO mnew = 2, matrix%m
          msofar = mnew - 1
          newclr = newdia + 1
          CALL SCU_triangular_solve( msofar, data%R( : newdia ),               &
                               data%R( newclr : newdia + msofar ), .TRUE. )
          newdia = newdia + mnew
          scalar = data%R( newdia ) -                                          &
                     DOT_PRODUCT( data%R( newclr : newclr + msofar - 1 ),      &
                                  data%R( newclr : newclr + msofar - 1 ) )

!  Check that the matrix is indeed positive definite

          IF ( scalar <= zero ) THEN 
            IF ( data%class == 3 ) THEN
               status = - 10
            ELSE
               status = - 11
            END IF
            RETURN
          END IF
          IF ( data%class == 3 ) THEN
            info%inertia( 1 ) = info%inertia( 1 ) + 1
          ELSE
            info%inertia( 2 ) = info%inertia( 2 ) + 1
          END IF
          data%R( newdia ) = sqrt( scalar )
        END DO
      END IF

 100  CONTINUE
      data%m = matrix%m
      data%got_factors = .TRUE.
      status = 0 ; RETURN

!  End of SCU_factorize

    END SUBROUTINE SCU_factorize

!-*-*-*-*-*-*-  - S C U _ s o l v e    S U B R O U T I N E   -*-*-*-*-*-*-*-*-

    SUBROUTINE SCU_solve( matrix, data, RHS, X, VECTOR, status )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the system of equations
!
!     / A  B \ / x1 \ _ / rhs1 \
!     \ C  D / \ x2 / ~ \ rhs2 /
!
!  Nick Gould, Fortran 77 version: August 3rd 1988
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( SCU_matrix_type ), INTENT( IN ) :: matrix
      TYPE ( SCU_data_type ), INTENT( INOUT ) :: data
      INTEGER, INTENT( INOUT ) :: status
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                               DIMENSION ( matrix%n + matrix%m ) :: RHS
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
                               DIMENSION ( matrix%n + matrix%m ) :: X
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION ( matrix%n ) :: VECTOR

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: icol, irow, j, k, np1, npm
      REAL ( KIND = wp ) :: scalar

!-----------------------------------------------
!   E x t e r n a l   F u n c t i o n s
!-----------------------------------------------

      IF ( status <= 0 ) THEN ; status = - 2 ; RETURN ; END IF

      np1 = matrix%n + 1 ; npm = matrix%n + matrix%m
      IF ( status == 1 ) data%jumpto = 1
      SELECT CASE ( data%jumpto )

!  Compute the part of the solution X2

      CASE ( 1 : 2 )

        IF ( data%jumpto == 1 ) THEN

!  Ensure that m is correct

          IF ( .NOT. data%got_factors ) THEN ; status = - 3 ; RETURN ; END IF
          IF ( data%m /= matrix%m ) THEN ; status = - 8 ; RETURN ; END IF

          X( np1 : npm ) = zero
          IF ( matrix%n > 0 ) THEN
            VECTOR = RHS( : matrix%n )

!  Return to get the product A(inverse) * VECTOR

            status = 2 ; data%jumpto = 2 ; RETURN
          END IF
        END IF

!  Copy VECTOR onto X1

        X( : matrix%n ) = VECTOR
        IF ( matrix%m > 0 ) THEN

!  Transform the right-hand side

          IF ( data%class == 4 ) THEN

!  S is negative definite and the factorization of -S is used.

            DO irow = 1, matrix%m
              scalar = - RHS( matrix%n + irow )
              IF ( matrix%n > 0 ) THEN
                DO k = matrix%BD_col_start( irow ),                            &
                       matrix%BD_col_start( irow + 1 ) - 1
                  j = matrix%BD_row( k )
                  IF ( j <= matrix%n )                                         &
                    scalar = scalar + matrix%BD_val( k ) * VECTOR( j )
                END DO
              END IF
              X( matrix%n + irow ) = scalar
            END DO
          ELSE

!  The factorization of S is used

            DO irow = 1, matrix%m
              scalar = RHS( matrix%n + irow )
              IF ( matrix%n > 0 ) THEN

!  S is symmetric

                IF ( data%class > 1 ) THEN
                  DO k = matrix%BD_col_start( irow ), &
                         matrix%BD_col_start( irow + 1 ) - 1
                    j = matrix%BD_row( k )
                    IF ( j <= matrix%n )                                       &
                      scalar = scalar - matrix%BD_val( k ) * VECTOR( j )
                  END DO
                ELSE

!  S is unsymmetric

                  DO k = matrix%CD_row_start( irow ),                          &
                         matrix%CD_row_start( irow + 1 ) - 1
                    j = matrix%CD_col( k )
                    IF ( j <= matrix%n )                                       &
                      scalar = scalar - matrix%CD_val( k ) * VECTOR( j )
                  END DO
                END IF
              END IF

!  If a QR factorization is used, transform by Q(transpose)

              IF ( data%class <= 2 ) THEN
                X( np1 : npm ) = X( np1 : npm ) &
                                   + scalar * data%Q( irow, : matrix%m )
              ELSE
                X( matrix%n + irow ) = scalar
              END IF
            END DO
          END IF
  
          IF ( data%class >= 3 )                                               &
            CALL SCU_triangular_solve( matrix%m, data%R, X( np1 : npm ),       &
                                        .TRUE. )

!  Transform by R(inverse)

          CALL SCU_triangular_solve( matrix%m, data%R, X( np1 : npm ),        &
                                      .FALSE. )

!  Compute the part of the solution X1

          IF ( matrix%n > 0 ) THEN
            VECTOR( : matrix%n ) = zero
            DO icol = 1, matrix%m
              scalar = X( matrix%n + icol )
              DO k = matrix%BD_col_start( icol ),                              &
                     matrix%BD_col_start( icol + 1 ) - 1
                j = matrix%BD_row( k )
                IF ( j <= matrix%n )                                           &
                  VECTOR( j ) = VECTOR( j ) + matrix%BD_val( k ) * scalar
              END DO
            END DO

!  Return to get the product A(inverse) * VECTOR

            status = 2 ; data%jumpto = 3 ; RETURN
          END IF
        END IF

      CASE ( 3 )
        X( : matrix%n ) = X( : matrix%n ) - VECTOR( : matrix%n )
      END SELECT
      status = 0 ; RETURN

!  End of SCU_solve

    END SUBROUTINE SCU_solve

!-*-*-*-*-*-*-*-  S C U _ a p p e n d    S U B R O U T I N E   -*-*-*-*-*-*-*-

    SUBROUTINE SCU_append( matrix, data, VECTOR, status, info )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Form and factorize the Schur complement
!
!      S = D - C * A(inverse) * B
!
!  of the matrix A in the symmetric or unsymmetric block matrix
!
!     / A  B \
!     \ C  D /
!
!  When an extra row and column are appended
!
!  Nick Gould, Fortran 77 version: August 3rd 1988
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( SCU_matrix_type ), INTENT( INOUT ) :: matrix
      TYPE ( SCU_info_type ), INTENT( INOUT ) :: info
      TYPE ( SCU_data_type ), INTENT( INOUT ) :: data
      INTEGER, INTENT( INOUT ) :: status
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
                               DIMENSION ( matrix%n ) :: VECTOR

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i, icol, irow, j, k, k1, newclr, kirn, kjcn, mnew,            &
                 sign_determinant
      REAL ( KIND = wp ) :: scalar

!-----------------------------------------------
!   E x t e r n a l   F u n c t i o n s
!-----------------------------------------------

      IF ( status <= 0 ) THEN ; status = - 2 ; RETURN ; END IF

      mnew = matrix%m + 1
      newclr = matrix%m * mnew / 2

      IF ( status == 2 ) GO TO 20
      IF ( status == 3 ) GO TO 30

!  The new row and column will be the mnew-th

      IF ( matrix%m < 0 .OR. mnew > data%m_max .OR. matrix%n < 0 ) THEN
           status = - 1 ; RETURN ; END IF

!  Ensure that array arguments are present when required

      IF ( .NOT. data%got_factors ) THEN ; status = - 3 ; RETURN ; END IF

      IF ( data%m /= matrix%m ) THEN ; status = - 8 ; RETURN ; END IF

!  Ensure that the matrices are large enough

      IF ( SIZE( matrix%BD_col_start ) < mnew + 1 ) THEN
        status = - 6 ; RETURN
      ELSE IF ( MIN( SIZE( matrix%BD_val ), SIZE( matrix%BD_row ) ) <          &
                   matrix%BD_col_start( mnew + 1 ) - 1 ) THEN
        status = - 6 ; RETURN
      END IF

!  Ensure that same is true for optional arrays

      IF ( data%class == 1 ) THEN
        IF ( SIZE( matrix%CD_row_start ) < mnew  + 1 ) THEN
           status = - 7 ; RETURN
        ELSE IF ( MIN( SIZE( matrix%CD_val ), SIZE( matrix%CD_col ) ) <        &
               matrix%CD_row_start( mnew + 1 ) - 1 ) THEN
           status = - 7 ; RETURN
        END IF
      END IF

!  Remove subdiagonal elements from the data structures associated
!  with the new row of B and the upper triangular part of D

      k1 = matrix%BD_col_start( mnew )
      kirn = k1
      data%newdia = matrix%n + mnew
      DO k = k1, matrix%BD_col_start( mnew + 1 ) - 1
        i = matrix%BD_row( k )
        IF ( i <= data%newdia ) THEN
          matrix%BD_row( kirn ) = i
          matrix%BD_val( kirn ) = matrix%BD_val( k )
          kirn = kirn + 1
        END IF
      END DO
      matrix%BD_col_start( mnew + 1 ) = kirn

!  Remove diagonal and superdiagonal elements from the data
!  structures associated with C and the lower triangular part of D

      IF ( data%class == 1 ) THEN
        k1 = matrix%CD_row_start( mnew )
        kjcn = k1
        DO k = k1, matrix%CD_row_start( mnew + 1 ) - 1
          j = matrix%CD_col( k )
          IF ( j < data%newdia ) THEN
            matrix%CD_col( kjcn ) = j
            matrix%CD_val( kjcn ) = matrix%CD_val( k )
            kjcn = kjcn + 1
          END IF
        END DO
        matrix%CD_row_start( mnew + 1 ) = kjcn
      END IF

!  Return to obtain the inverse of A times the first n components
!  of colnew. First calculate colnew

      data%R( newclr + 1 : newclr + matrix%m ) = zero
      data%W( : mnew ) = zero
      VECTOR( : matrix%n ) = zero

!  S is symmetric

      DO k = matrix%BD_col_start( mnew ), matrix%BD_col_start( mnew + 1 ) - 1
        j = matrix%BD_row( k )
        IF ( j <= matrix%n ) THEN
          VECTOR( j ) = matrix%BD_val( k )
        ELSE
          data%W( j - matrix%n ) = matrix%BD_val( k )
        END IF
      END DO
      data%dianew = data%W( mnew )

      IF ( matrix%n > 0 ) THEN ; status = 2 ; RETURN ; END IF

!  Form the new last column of S

 20   CONTINUE
      IF ( matrix%m > 0 ) THEN

        IF ( data%class == 4 ) THEN

!  S is negative definite and the factors of -S are used

          DO irow = 1, matrix%m
            scalar = - data%W( irow )
            IF ( matrix%n > 0 ) THEN
              DO k = matrix%BD_col_start( irow ),                              &
                     matrix%BD_col_start( irow + 1 ) - 1
                j = matrix%BD_row( k )
                IF ( j <= matrix%n )                                           &
                   scalar = scalar + matrix%BD_val( k ) * VECTOR( j )
              END DO
            END IF

!  Transform the new column of R

            data%R( newclr + irow ) = scalar
          END DO

        ELSE

!  The factors of S are used

          DO irow = 1, matrix%m
            scalar = data%W( irow )
            IF ( matrix%n > 0 ) THEN

!  S is symmetric

              IF ( data%class > 1 ) THEN
                DO k = matrix%BD_col_start( irow ),                            &
                       matrix%BD_col_start( irow + 1 ) - 1
                  j = matrix%BD_row( k )
                  IF ( j <= matrix%n )                                         &
                     scalar = scalar - matrix%BD_val( k ) * VECTOR( j )
                END DO

!  S is unsymmetric

              ELSE
                DO k = matrix%CD_row_start( irow ),                            &
                       matrix%CD_row_start( irow + 1 ) - 1
                  j = matrix%CD_col( k )
                  IF ( j <= matrix%n )                                         &
                     scalar = scalar - matrix%CD_val( k ) * VECTOR( j )
                END DO
              END IF
            END IF

!  Transform the new column of R

            IF ( data%class <= 2 ) THEN
              IF ( data%class == 2 ) data%W( irow ) = scalar
              data%R( newclr + 1 : newclr + matrix%m ) =                       &
                data%R( newclr + 1 : newclr + matrix%m )                       &
                 + scalar * data%Q( irow, 1 : matrix%m )

!  Initialize the new row and column of Q

              data%Q( : matrix%m, mnew ) = zero
              data%Q( mnew, : matrix%m ) = zero
            ELSE
              data%R( newclr + irow ) = scalar
            END IF

          END DO
        END IF
      END IF

!  If the matrix is unsymmetric the new row is also needed

      IF ( data%class > 1 ) GO TO 40

!  Set up rownew (in VECTOR).

      data%W( : matrix%m ) = zero
      VECTOR( : matrix%n ) = zero
      DO k = matrix%CD_row_start( mnew ), matrix%CD_row_start( mnew + 1 ) - 1
        j = matrix%CD_col( k )
        IF ( j <= matrix%n ) THEN
          VECTOR( j ) = matrix%CD_val( k )
        ELSE
          data%W( j - matrix%n ) = matrix%CD_val( k )
        END IF
      END DO

!  Return to obtain the inverse of A(transpose) times the first n
!  components of rownew

      IF ( matrix%n > 0 ) THEN ; status = 3 ; RETURN ; END IF

!  Form the new last row of S

 30   CONTINUE
      IF ( matrix%m > 0 .AND. matrix%n > 0 ) THEN
        DO icol = 1, matrix%m
          scalar = data%W( icol )
          DO k = matrix%BD_col_start( icol ),                                  &
                 matrix%BD_col_start( icol + 1 ) - 1
            j = matrix%BD_row( k )
            IF ( j <= matrix%n )                                               &
               scalar = scalar - matrix%BD_val( k ) * VECTOR( j )
          END DO
          data%W( icol ) = scalar
        END DO
      END IF

!  Form the new diagonal of S prior to the R(trans) R factorization

 40   CONTINUE
      IF ( matrix%n > 0 ) THEN
        DO k = matrix%BD_col_start( mnew ), matrix%BD_col_start( mnew + 1 ) - 1
          j = matrix%BD_row( k )
          IF ( j <= matrix%n )                                                 &
             data%dianew = data%dianew - matrix%BD_val( k ) * VECTOR( j )
        END DO
      END IF

      IF ( data%class >= 3 ) THEN
        IF ( data%class == 4 ) THEN
          scalar = - data%dianew
        ELSE
          scalar = data%dianew
        END IF

!  Find the new column of R

        IF ( matrix%m > 0 ) THEN
          CALL SCU_triangular_solve( matrix%m, data%R( : newclr ),            &
                    data%R( newclr + 1 : newclr + matrix%m ), .TRUE. )
!         IF ( data%class == 4 ) THEN
!           scalar = scalar                                                    &
!                  - DOT_PRODUCT( data%R( newclr + 1 : newclr + matrix%m ),    &
!                                 data%R( newclr + 1 : newclr + matrix%m ) )
!         ELSE
            scalar = scalar                                                    &
                   - DOT_PRODUCT( data%R( newclr + 1 : newclr + matrix%m ),    &
                                  data%R( newclr + 1 : newclr + matrix%m ) )
!         END IF
        END IF

!  Check that the matrix is indeed positive definite

        IF ( scalar <= zero ) THEN 
          IF ( data%class == 3 ) THEN
             status = - 10
          ELSE
             status = - 11
          END IF
          RETURN
        END IF
        IF ( data%class == 3 ) THEN
          info%inertia( 1 ) = info%inertia( 1 ) + 1
        ELSE
          info%inertia( 2 ) = info%inertia( 2 ) + 1
        END IF

!  Find the new diagonal of R

        data%R( mnew * ( mnew + 1 ) / 2 ) = sqrt( scalar )
      ELSE

!  Form the new diagonal of S prior to the QR factorization

        data%W( mnew ) = data%dianew

!  Set the new diagonal entry of Q

        data%Q( mnew, mnew ) = one

!  Reduce the new row of S to zero by applying plane-rotation matrices

        CALL SCU_triangular( matrix%m, 1, data%R, data%W, status,             &
                              Q = data%Q )
        IF ( status < 0 ) THEN
          info%inertia( 2 ) = info%inertia( 2 ) + 1
          data%sign_determinant = - data%sign_determinant
          data%m = data%m + 1
          matrix%m = data%m
          RETURN
        END IF

!  Determine the inertia of S

        IF ( data%class == 2 ) THEN
          sign_determinant = SCU_sign_determinant( mnew, data%R )
          IF ( sign_determinant == data%sign_determinant ) THEN
            info%inertia( 1 ) = info%inertia( 1 ) + 1
          ELSE
            info%inertia( 2 ) = info%inertia( 2 ) + 1
            data%sign_determinant = sign_determinant
          END IF
        END IF

      END IF
!     write(6,"( 8ES10.2 )" ) ( data%R( ( i * ( i + 1 ) ) / 2 ), i = 1, mnew )

      data%m = data%m + 1
      matrix%m = data%m
      status = 0 ; RETURN

!  End of SCU_append

    END SUBROUTINE SCU_append

!-*-*-*-*-*-*-*-  S C U _ d e l e t e    S U B R O U T I N E   -*-*-*-*-*-*-*-

    SUBROUTINE SCU_delete( matrix, data, VECTOR, status, info, col_del,      &
                            row_del )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Reorder the data structures and update the factorization of the
!  Schur complement matrix
!
!      S = D - C * A(inverse) * B
!
!  of the matrix A in the symmetric or unsymmetric block matrix
!
!     / A  B \
!     \ C  D /
!
!  when row row_del of (C  D) and column col_del of (B over D)
!  (row_del = col_del when class > 1) are deleted
!
!  Nick Gould, Fortran 77 version: January 5th 1989
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( SCU_matrix_type ), INTENT( INOUT ) :: matrix
      TYPE ( SCU_info_type ), INTENT( INOUT ) :: info
      TYPE ( SCU_data_type ), INTENT( INOUT ) :: data
      INTEGER, INTENT( IN ) :: col_del
      INTEGER, INTENT( IN ), OPTIONAL :: row_del
      INTEGER, INTENT( OUT ) :: status
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION ( matrix%n ) :: VECTOR

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i, idrop, irow, j, jcol, k, k2, kirn, kjcn, njcol, mm1, mp1, &
              njdrop, irow2, jcol2, nirow, last, nidrop, newlst, nextr, row,  &
              sign_determinant
      REAL ( KIND = wp ) :: c, s, x, y
      LOGICAL :: qused

!  Ensure that optional arguments are present when required

      IF ( matrix%m < 0 .OR. matrix%m > data%m_max .OR. matrix%n < 0 .OR.      &
           col_del < 1 .OR. col_del > matrix%m ) THEN
        status = - 1 ; RETURN ; END IF

      mp1 = matrix%m + 1
      IF ( PRESENT( row_del ) .AND. data%class == 1 ) THEN
         row = row_del
      ELSE
         row = col_del
      END IF

!  Ensure that array arguments are present when required

      IF ( .NOT. data%got_factors ) THEN ; status = - 3 ; RETURN ; END IF

!  Ensure that the optional arrays are large enough

      IF ( data%class == 1 ) THEN
        IF ( row < 1 .OR. row > matrix%m )                                     &
             THEN ; status = - 1 ; RETURN ; END IF

        IF ( SIZE( matrix%CD_row_start ) < mp1 ) THEN
           status = - 7 ; RETURN
        ELSE IF ( MIN( SIZE( matrix%CD_val ), SIZE( matrix%CD_col ) ) <        &
            matrix%CD_row_start( mp1 ) + abs( row - col_del ) - 1 )            &
           THEN ; status = - 7 ; RETURN
        END IF
      END IF

      IF ( data%m /= matrix%m ) THEN ; status = - 8 ; RETURN ; END IF

      mm1 = matrix%m - 1
      njdrop = matrix%n + col_del

!  Update the data structures to allow for the row and column removal

      IF ( data%class == 1 ) THEN

!  Consider unsymmetric matrices

        nidrop = matrix%n + row

!  Update the data structures when row is less than or equal to col_del

        IF ( row <= col_del ) THEN
          last = matrix%BD_col_start( mp1 ) + col_del - row
          IF ( row < col_del ) THEN

!  Insert gaps to accomodate incoming subdiagonal elements in the
!  superdiagonal storage scheme

            DO j = matrix%m, row, - 1
              k2 = matrix%BD_col_start( j + 1 ) - 1
              matrix%BD_col_start( j + 1 ) = last
              IF ( j < col_del ) THEN
                last = last - 1
                matrix%BD_row( last ) = 0
              END IF
              DO k = k2, matrix%BD_col_start( j ), - 1
                last = last - 1
                matrix%BD_row( last ) = matrix%BD_row( k )
                matrix%BD_val( last ) = matrix%BD_val( k )
              END DO
            END DO
            matrix%BD_col_start( row ) = last
          END IF
      
          jcol = row
          jcol2 = jcol
          kirn = matrix%BD_col_start( row )
          kjcn = matrix%CD_row_start( row )
          DO irow = row, mm1
            irow2 = irow + 1
            IF ( jcol2 == col_del ) jcol2 = jcol2 + 1

!  Search the current column, jcol, to remove any entry in row row

            matrix%BD_col_start( jcol ) = kirn
            DO k = matrix%BD_col_start( jcol2 ),                               &
                   matrix%BD_col_start( jcol2 + 1 ) - 1
              i = matrix%BD_row( k )
              IF ( i /= nidrop .AND. i /= 0 ) THEN
                IF ( i > nidrop ) THEN
                  matrix%BD_row( kirn ) = i - 1
                ELSE
                  matrix%BD_row( kirn ) = i
                END IF
                matrix%BD_val( kirn ) = matrix%BD_val( k )
                kirn = kirn + 1
              END IF
            END DO

!  Search the current row, irow, to see if there is any entry in column njcol

            njcol = matrix%n + jcol2
            matrix%CD_row_start( irow ) = kjcn
            DO k = matrix%CD_row_start( irow2 ),                               &
                   matrix%CD_row_start( irow2 + 1 ) - 1
              j = matrix%CD_col( k )
              IF ( j /= njdrop ) THEN
                IF ( j /= njcol ) THEN
                  IF ( j > njdrop ) THEN
                    matrix%CD_col( kjcn ) = j - 1
                  ELSE
                    matrix%CD_col( kjcn ) = j
                  END IF
                  matrix%CD_val( kjcn ) = matrix%CD_val( k )
                  kjcn = kjcn + 1
                ELSE

!  If there is a term which was previously in the lower triangular part which 
!  should now be in the upper triangle, move it to its appropriate position

                  matrix%BD_row( kirn ) = matrix%n + jcol
                  matrix%BD_val( kirn ) = matrix%CD_val( k )
                  kirn = kirn + 1
                END IF
              END IF
            END DO
            jcol = jcol + 1
            jcol2 = jcol2 + 1
          END DO
        ELSE
          last = matrix%CD_row_start( mp1 ) + row - col_del - 1

!  Insert gaps to accomodate incoming superdiagonal elements in the
!  subdiagonal storage scheme

          DO j = matrix%m, col_del + 1, - 1
            matrix%CD_row_start( j + 1 ) = last
            IF ( j < row ) THEN
              last = last - 1
              matrix%CD_col( last ) = 0
            END IF
            DO k = matrix%CD_row_start( j + 1 ) - 1,                           &
                   matrix%CD_row_start( j ), - 1
              last = last - 1
              matrix%CD_col( last ) = matrix%CD_col( k )
              matrix%CD_val( last ) = matrix%CD_val( k )
            END DO
          END DO
          irow = col_del + 1
          matrix%CD_row_start( irow ) = last
      
          irow2 = irow
          kirn = matrix%BD_col_start( col_del )
          kjcn = matrix%CD_row_start( irow )
          DO jcol = col_del, mm1
            jcol2 = jcol + 1

!  Search the current row, irow, to remove any entry in column col_del

            IF ( irow2 /= row ) THEN
              matrix%CD_row_start( irow ) = kjcn
              DO k = matrix%CD_row_start( irow2 ),                             &
                     matrix%CD_row_start( irow2 + 1 ) - 1
                j = matrix%CD_col( k )
                IF ( j /= njdrop .AND. j /= 0 ) THEN
                  IF ( j > njdrop ) THEN
                    matrix%CD_col( kjcn ) = j - 1
                  ELSE
                    matrix%CD_col( kjcn ) = j
                  END IF
                  matrix%CD_val( kjcn ) = matrix%CD_val( k )
                  kjcn = kjcn + 1
                END IF
              END DO
            END IF

!  Search the current column, jcol, to see if there is any entry in row nirow

            nirow = matrix%n + jcol2 + jcol - irow + 1
            matrix%BD_col_start( jcol ) = kirn
            DO k = matrix%BD_col_start( jcol2 ),                               &
                   matrix%BD_col_start( jcol2 + 1 ) - 1
              i = matrix%BD_row( k )
              IF ( i /= nidrop ) THEN
                IF ( i /= nirow ) THEN
                  IF ( i > nidrop ) THEN
                    matrix%BD_row( kirn ) = i - 1
                  ELSE
                    matrix%BD_row( kirn ) = i
                  END IF
                  matrix%BD_val( kirn ) = matrix%BD_val( k )
                  kirn = kirn + 1
                ELSE

!  If there is a term which was previously in the upper triangular part which 
!  should now be in the lower triangle, move it to its appropriate position

                  matrix%CD_col( kjcn ) = matrix%n + irow - 1
                  matrix%CD_val( kjcn ) = matrix%BD_val( k )
                  kjcn = kjcn + 1
                END IF
              END IF
            END DO
            IF ( irow2 /= row ) irow = irow + 1
            irow2 = irow2 + 1
          END DO
      
        END IF
        matrix%CD_row_start( matrix%m ) = kjcn

!  Consider symmetric matrices

      ELSE
        kirn = matrix%BD_col_start( col_del )

!  Search each column to remove entries in row col_del

        DO jcol = col_del, mm1
          jcol2 = jcol + 1
          matrix%BD_col_start( jcol ) = kirn
          DO k = matrix%BD_col_start( jcol2 ),                                 &
                 matrix%BD_col_start( jcol2 + 1 ) - 1
            i = matrix%BD_row( k )
            IF ( i /= njdrop ) THEN
              IF ( i > njdrop ) THEN
                matrix%BD_row( kirn ) = i - 1
              ELSE
                matrix%BD_row( kirn ) = i
              END IF
              matrix%BD_val( kirn ) = matrix%BD_val( k )
              kirn = kirn + 1
            END IF
          END DO
        END DO
      END IF
      matrix%BD_col_start( matrix%m ) = kirn

!  Calculate the appropriate factorization of the reduced Schur complement

      qused = data%class <= 2

      IF ( qused ) THEN

        IF ( data%class == 1 ) THEN
          idrop = row
        ELSE
          idrop = col_del
        END IF

!  If a QR factorization is used, shuffle row idrop of Q to the end

        IF ( idrop < matrix%m ) THEN
          VECTOR( : matrix%m ) = data%Q( idrop , : matrix%m )
          DO i = idrop, mm1
            data%Q( i, : matrix%m ) = data%Q( i + 1, : matrix%m )
          END DO
          data%Q( matrix%m, : matrix%m ) = VECTOR( : matrix%m )
        END IF

!  Shuffle column col_del to the end of Q

        IF ( col_del < matrix%m ) THEN
          VECTOR( : matrix%m ) = data%Q( : matrix%m, col_del )
          DO j = col_del, mm1
            data%Q( : matrix%m, j ) = data%Q( : matrix%m, j + 1 )
          END DO
          data%Q( : matrix%m, matrix%m ) = VECTOR( : matrix%m )
        END IF
      END IF

      data%W( : col_del - 1 ) = zero
      VECTOR( col_del : matrix%m - 1 ) = zero

!  Shuffle row and column col_del to the end of R. Store the
!  new last row of R in W, and the new last column in VECTOR.

      IF ( col_del < matrix%m ) THEN
        last = col_del * ( col_del - 1 ) / 2 + 1
        newlst = last

        IF ( col_del - 1 > 0 ) THEN
          VECTOR( : col_del - 1 ) = data%R( last : col_del - 2 + last )
          last = col_del - 1 + last
        END IF

        data%W( matrix%m ) = data%R( last )
        VECTOR( matrix%m ) = data%R( last )

        last = last + 1
        DO j = col_del + 1, matrix%m
          DO i = 1, j
            IF ( i == col_del ) THEN
              data%W( j - 1 ) = data%R( last )
            ELSE
              data%R( newlst ) = data%R( last )
              newlst = newlst + 1
            END IF
            last = last + 1
          END DO
        END DO

        data%R( newlst : newlst + matrix%m - 1 ) = VECTOR( : matrix%m )

!  Restore R to upper triangular form by applying a sequence of
!  plane-rotation matrices

        IF ( qused ) THEN
          CALL SCU_triangular( mm1, col_del, data%R, data%W, status,          &
                               Q = data%Q )
        ELSE
          CALL SCU_triangular( mm1, col_del, data%R, data%W, status )
        END IF

      END IF

      IF ( data%class == 3 ) THEN
        info%inertia( 1 ) = info%inertia( 1 ) - 1
      ELSE IF ( data%class == 4 ) THEN
        info%inertia( 2 ) = info%inertia( 2 ) - 1
      END IF

      IF ( qused ) THEN

!  We now have the QR factorization of the permuted matrix. It remains to apply
!  further plane rotation matrices to make the first (m-1)*(m-1) submatrix of Q
!  orthogonal. Reduce the last row of Q to the last row of the identity matrix
!  by applying an appropriate sequence of plane rotations. (This forces the
!  last column of Q to be of the same form)

        DO j = mm1, 1, - 1

!  Apply a plane rotation to eliminate the entry in column j using
!  the entry in column m

          CALL ROTG( data%Q( matrix%m, matrix%m ),                             &
                     data%Q( matrix%m, j ), c, s )
          data%Q( matrix%m, j ) = zero

!  Apply the rotation to the remaining entries in columns j and m

          CALL ROT( mm1, data%Q( 1 : mm1, matrix%m ), 1,                       &
                    data%Q( 1 : mm1, j ), 1, c, s )

!  Apply the rotation to the entries in rows j and m of R

          nextr = j * ( j + 1 ) / 2
          data%W( j ) = zero
          DO k = j, mm1
            x = data%W( k )
            y = data%R( nextr )
            data%W( k ) = c * x + s * y
            data%R( nextr ) = c * y - s * x
            nextr = nextr + k
          END DO

        END DO

!  Check that the new Schur complement is nonsingular

        DO j = 1, mm1
          IF ( abs( data%R( j * ( j + 1 ) / 2 ) ) <= epsmch ) THEN
!           write(6,*) ' new diag = ', abs( data%R( j * ( j + 1 ) / 2 ) )
            status = - 9 ; RETURN
          END IF
        END DO

!  Determine the inertia of S

!       WRITE(6,"( ' diag(Q) removed = ', ES12.4)") data%Q( matrix%m, matrix%m )
        IF ( data%Q( matrix%m, matrix%m ) < zero )                             &
          data%sign_determinant = - data%sign_determinant
        IF ( data%class == 2 ) THEN
          sign_determinant = SCU_sign_determinant( mm1, data%R )
          IF ( sign_determinant == data%sign_determinant ) THEN
            info%inertia( 1 ) = info%inertia( 1 ) - 1
          ELSE
            info%inertia( 2 ) = info%inertia( 2 ) - 1
            data%sign_determinant = sign_determinant
          END IF
        END IF

      END IF

      data%m = data%m - 1
      matrix%m = data%m
      status = 0 ; RETURN

!  END OF SCU_delete

    END SUBROUTINE SCU_delete

!-*-*-*-*-*-*-  S C U _ i n c r e a s e   S U B R O U T I N E   -*-*-*-*-*-*-

    SUBROUTINE SCU_increase_diagonal( data, diagonal, info )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Adds a diagonal term to the last added row/column to change the inertia
!  of the Schur complement S = D - C * A(inverse) * B
!
!  Nick Gould, May 21st 1999
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( SCU_info_type ), INTENT( INOUT ) :: info
      TYPE ( SCU_data_type ), INTENT( INOUT ) :: data
      REAL ( KIND = wp ), INTENT( OUT ) :: diagonal

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: m, rstart, rend
      m = data%m
      rend = m * ( m + 1 ) / 2
      rstart = rend - m + 1

!     write(6,"('h,r', 2ES12.4)") data%Q( m, m ), data%R( rend )

!     diagonal = - 2.0_wp * ( data%R( rend ) / data%Q( m, m ) )
      diagonal = MAX( - 2.0_wp * ( data%R( rend ) / data%Q( m, m ) ),          &
                      - ( data%R( rend ) / data%Q( m, m ) ) +                  &
                          r_pos / ABS( data%Q( m, m ) ) )
!     diagonal = ( MAX( - data%R( rend ), SIGN( r_pos, data%Q( m, m ) ) ) -    &
!                  data%R( rend ) ) / data%Q( m, m )
!     write(6,"('increase diagonal by ', ES12.4)") diagonal
      
!     write(6,*) ' rstart, rend ', rstart, rend
!     write(6,"( ' R ', /, (5ES12.4) )" ) data%R( rstart : rend ) 
!     write(6,"( ' Q ', /, (5ES12.4) )" ) data%Q( m , : m )
      data%R( rstart : rend ) = data%R( rstart : rend ) +                      &
                                diagonal * data%Q( m , : m )
      
      info%inertia( 1 ) = info%inertia( 1 ) + 1
      info%inertia( 2 ) = info%inertia( 2 ) - 1
      data%sign_determinant = - data%sign_determinant

      RETURN

    END SUBROUTINE SCU_increase_diagonal

!-*-*-*-*-*- S C U _ t e r m i n a t e    S U B R O U T I N E   -*-*-*-*-*-*-

    SUBROUTINE SCU_terminate( data, status, info )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Remove the internal data structures
!
!  Nick Gould, April 14th 1999
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( SCU_info_type ), INTENT( OUT ) :: info
      TYPE ( SCU_data_type ), INTENT( INOUT ) :: data
      INTEGER, INTENT( OUT ) :: status

!  Allocate the data arrays Q, R and W

      status = 0 ; info%alloc_status = 0
      IF ( data%class <= 2 ) THEN
        IF ( ALLOCATED( data%Q ) ) DEALLOCATE( data%Q, STAT = status )
        IF ( status /= 0 ) info%alloc_status = status
      END IF

      IF ( ALLOCATED( data%R ) ) DEALLOCATE( data%R, STAT = status )
      IF ( status /= 0 ) info%alloc_status = status

      IF ( ALLOCATED( data%W ) ) DEALLOCATE( data%W, STAT = status )
      IF ( status /= 0 ) info%alloc_status = status

      IF ( info%alloc_status /= 0 ) status = - 12
      RETURN

    END SUBROUTINE SCU_terminate

!-*-*-*-*-*-   S C U _ t r i a n g u l a r   S U B R O U T I N E   -*-*-*-*-*-

    SUBROUTINE SCU_triangular( msofar, jbegin, R, SPIKE, status, Q )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Use plane-rotation matrices to reduce an upper triangular matrix with a 
!  new horizontal spike row to upper triangular form. The spike is input in 
!  the array SPIKE and its first nonzero occurs in position jbegin.
!
!  Nick Gould, Fortran 77 version: August 5th 1988
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: msofar, jbegin
      INTEGER, INTENT( OUT ) :: status
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION ( : ) :: R
      REAL ( KIND = wp ), INTENT( INOUT ), OPTIONAL,                      &
                               DIMENSION ( : , : ) :: Q
      REAL ( KIND = wp ), INTENT( INOUT ),                                &
                               DIMENSION ( msofar + 1 ) :: SPIKE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: j, k, mnew, nextr, nextw
      REAL ( KIND = wp ) :: c, s, x, y

      mnew = msofar + 1

!  Reduce the new row to zero by applying plane-rotation matrices

!     write(6,"('spike',/,(5ES12.4))")  SPIKE
!     write(6,*) ' jbegin, msofar ', jbegin, msofar
      IF ( jbegin <= msofar ) THEN

        DO j = jbegin, msofar
          nextw = j + 1
          nextr = j * nextw / 2

!  Use a plane-rotation in the plane (j,mnew)

          CALL ROTG( R( nextr ), SPIKE( j ), c, s )

!  Apply the plane-rotations to the remaining elements in rows j and mnew of R

          nextr = nextr + j
          DO k = j + 1, mnew
            x = R( nextr )
            y = SPIKE( k )
            R( nextr ) = c * x + s * y
            SPIKE( k ) = c * y - s * x
            nextr = nextr + k
          END DO

!  Apply the plane-rotations to the remaining elements in columns
!  j and mnew of Q

!  NB: ROT replaced by do loop to prevent mkl bug ... sigh
!         IF ( PRESENT( Q ) )                                                  &
!           CALL ROT( mnew, Q( : mnew, j ), 1, Q( : mnew, mnew ), 1, c, s )
          IF ( PRESENT( Q ) ) THEN
            DO k = 1, mnew
              y = c * Q( k, j ) + s * Q( k, mnew )
              Q( k, mnew ) = c * Q( k, mnew ) - s * Q( k, j )
              Q( k, j ) = y
            END DO
          END IF
        END DO
      END IF

!  Check that the new diagonal entry of R is non-zero

      R( mnew * ( mnew + 1 ) / 2 ) = SPIKE( mnew )
!     write(6,*) ' new diag = ', abs( SPIKE( mnew ) )
      IF ( ABS( SPIKE( mnew ) ) > epsmch ) THEN
        status = 0
      ELSE
        status = - 9
      END IF

      RETURN

!  END OF SCU_triangular

    END SUBROUTINE SCU_triangular

!-*-*-  S C U _ t r i a n g u l a r _ s o l v e   S U B R O U T I N E  -*-*-*-

    SUBROUTINE SCU_triangular_solve( nr, R, X, trans )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the solution to the triangular systems
!
!     R * X(output) = X(input)               (trans = .FALSE.)
!
!  or 
!
!     R(transpose) * X(output) = X(input)    (trans = .TRUE.),
!
!  where R is an upper triangular matrix stored by columns
!
!  Nick Gould, Fortran 77 version: August 15th 1988
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: nr
      LOGICAL, INTENT( IN ) :: trans
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION ( : ) :: R
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION ( nr ) :: X

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i, nextr
      REAL ( KIND = wp ) :: scalar

!-----------------------------------------------
!   E x t e r n a l   F u n c t i o n s
!-----------------------------------------------

      IF ( nr <= 0 ) RETURN
      IF ( trans ) THEN

!  Solve R(transpose) * X(output) = X(input)

        X( 1 ) = X( 1 ) / R( 1 )

        IF ( nr > 1 ) THEN
          nextr = 2
          DO i = 2, nr
            scalar = X( i ) - DOT_PRODUCT( R( nextr : nextr + i - 2 ),         &
                                           X( : i - 1 ) )
            nextr = nextr + i
            X( i ) = scalar / R( nextr - 1 )
          END DO
        END IF

      ELSE

!  Solve R * X(output) = X(input)

        nextr = nr * ( nr + 1 ) / 2
        DO i = nr, 1, - 1
          scalar = X( i ) / R( nextr )
          X( i ) = scalar
          nextr = nextr - i
          X( : i - 1 ) = X( : i - 1 ) - scalar * R( nextr + 1 : nextr + i - 1 )
        END DO
      END IF

      RETURN

!  End of SCU_triangular_solve

    END SUBROUTINE SCU_triangular_solve

!-*-*-*-  S C U _ s i g n _ d e t e r m i n a n t    F U N C T I O N  -*-*-*-

    INTEGER FUNCTION SCU_sign_determinant( nr, R )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Find the sign of the determinant of R
!
!  Nick Gould, April 28th 1999
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: nr
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION ( : ) :: R

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i, diag
      REAL ( KIND = wp ) :: scalar

!-----------------------------------------------
!   E x t e r n a l   F u n c t i o n s
!-----------------------------------------------

      IF ( nr <= 0 ) THEN
         SCU_sign_determinant = 1
      ELSE
        diag = 0
        scalar = one
        DO i = 1, nr
          diag = diag + i
          scalar = SIGN( one, R( diag ) ) * scalar
        END DO

        IF ( scalar > zero ) THEN
          SCU_sign_determinant = 1
        ELSE
          SCU_sign_determinant = - 1
        END IF
      END IF

      RETURN

!  End of SCU_sign_determinant

    END FUNCTION SCU_sign_determinant

!  End of module GALAHAD_SCU_double

  END MODULE GALAHAD_SCU_double
