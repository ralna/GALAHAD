! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
! Updated 29/11/2002: extra arguments introduced to allow changing bandwidth

!-*-*-*-*-*-  L A N C E L O T  -B-  BAND   M O D U L E  *-*-*-*-*-*-*-*

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  January 23rd 1995

   MODULE LANCELOT_BAND_double

     IMPLICIT NONE
     
     PRIVATE
     PUBLIC :: BAND_factor, BAND_solve

!!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp

   CONTAINS

!-*-*-*-  L A N C E L O T  -B-  BAND_factor   S U B R O U T I N E   -*-*-*-*

     SUBROUTINE BAND_factor( n, nsemib, DIAG, OFFDIA, lsemib, status, PERT,    &
                             n_pert )

!  Compute the L D L(transpose) factorization of a banded matrix
!  A, modifying the diagonal D to ensure that the factorization succeeds

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, nsemib, lsemib
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION ( n ) :: DIAG
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION ( lsemib, n ) :: OFFDIA
     REAL ( KIND = wp ), OPTIONAL, INTENT( OUT ), DIMENSION ( n ) :: PERT
     INTEGER, OPTIONAL, INTENT( OUT ) :: n_pert
     
!-------------------------------------------
!   L a l   V a r i a b l e s
!-------------------------------------------
     
     INTEGER :: i, ipjm1, j, k, m
     REAL ( KIND = wp ) :: offd, tau1, tau2, gamma, offsum
     LOGICAL :: phase1
     
     IF ( nsemib <= lsemib ) THEN
       status = 0
     ELSE
       status = 1 ; RETURN
     END IF

     phase1 = .TRUE. ; gamma = one
     IF ( PRESENT( PERT ) ) PERT = zero
     IF ( PRESENT( n_pert ) ) n_pert = 0

!  Check that the initial diagonals are positive

     DO i = 1, n

!  The matrix is indefinite. Enter phase-2

       IF ( DIAG( i ) <= zero ) phase1 = .FALSE.

!  Find the largest diagonal entry

       gamma = MAX( gamma, ABS( DIAG( i ) ) )
     END DO

!  Set pivot tolerances

     tau1 = gamma * EPSILON( one ) ** 0.333 ; tau2 = tau1

!  Loop over the columns of the matrix

     DO i = 1, n
       m = MIN( nsemib, n - i )

!  Perform the I-th elimination. First, check that the resulting
!  pivots are positive

       IF ( phase1 ) THEN
         DO j = 1, m
           IF ( DIAG( i + j ) - ( OFFDIA( j, i ) / DIAG( i ) )                 &
             * OFFDIA( j, i ) <= tau1 ) THEN

!  The matrix is indefinite. Enter phase-2

             phase1 = .FALSE.
             GO TO 110
           END IF
         END DO
       END IF

!  If the matrix is indefinite, modify the diagonals

 110   CONTINUE

!  Compute the Gershgorin radius for the principal diagonal of the
!  last N - I by N - I submatrix

       IF ( .NOT. phase1 ) THEN
!         offsum = SUM( ABS( OFFDIA( : m, i ) ) )
          offsum = zero
          DO j = 1, m ; offsum = offsum + ABS( OFFDIA( j, i ) ) ; END DO

!  Perturb the diagonal so that the Gershgorin disk lies in the
!  positive half of the complex plane

          offsum = MAX( zero, - DIAG( i ) + MAX( offsum, tau2 ) )
!         WRITE(6,*) ' DIAGONAL ', I, ' MODIFIED BY ', OFFSUM
          DIAG( i ) = DIAG( i ) + offsum
          IF ( PRESENT( PERT ) ) PERT( i ) = offsum
          IF ( PRESENT( n_pert ) ) n_pert = n_pert + 1
       END IF

!  Perform the I-th step of the factorization

       DO j = 1, m
          offd = OFFDIA( j, i )

!  Update the Schur complement. (1) off diagonal terms

          ipjm1 = j
          DO k = 1, j - 1
            OFFDIA( ipjm1 - k, i + k ) =                                       &
              OFFDIA( ipjm1 - k, i + k ) - offd * OFFDIA( k, i )
          END DO

!  (2) diagonal terms

          offd = offd / DIAG( i )
          DIAG( i + j ) = DIAG( i + j ) - offd * OFFDIA( j, i )

!  Find the subdiagonal of the I-th column of the factor L.

          OFFDIA( j, i ) = offd
       END DO
     END DO
     RETURN

!  End of subroutine BAND_factor

     END SUBROUTINE BAND_factor

!-*-*-*-  L A N C E L O T  -B-  BAND_solve   S U B R O U T I N E   -*-*-*-*

     SUBROUTINE BAND_solve( n, nsemib, DIAG, OFFDIA, lsemib, RHS, status )

!  Solve the system of banded linear eqautions L D L(transpose) x = rhs,
!  putting the solution in RHS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, nsemib, lsemib
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION ( n ) :: DIAG
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION ( n ) :: RHS
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION ( lsemib, n ) :: OFFDIA

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, m
     REAL ( KIND = wp ) :: temp

     IF ( nsemib <= lsemib ) THEN
       status = 0
     ELSE
       status = 1 ; RETURN
     END IF

!  Forward solve to obtain the solution to L y = rhs, putting the
!  solution in RHS

     DO i = 1, n
       m = MIN( nsemib, n - i )
       temp = RHS( i )
       DO j = 1, m
         RHS( i + j ) = RHS( i + j ) - OFFDIA( j, i ) * temp
       END DO
!      RHS( i + 1 : i + m ) = RHS( i + 1 : i + m ) - OFFDIA( : m, i ) * temp

!  Obtain the solution to the diagonal system D y = rhs, putting the
!  solution in RHS

       RHS( i ) = temp / DIAG( i )
     END DO

!  Back solve to obtain the solution to L(TRANSPOSE) y = rhs,
!  putting the solution IN RHS

     DO i = n, 1, - 1
       m = MIN( nsemib, n - i )
!      RHS( i ) = RHS( i ) - SUM( OFFDIA( : m, i ) * RHS( i + 1 : i + m ) )
       temp = RHS( i )
       DO j = 1, m
         temp = temp - OFFDIA( j, i ) *  RHS( i + j )
       END DO
       RHS( i ) = temp
     END DO
     RETURN

!  End of subroutine BAND_solve

     END SUBROUTINE BAND_solve

!  End of module LANCELOT_BAND

   END MODULE LANCELOT_BAND_double
