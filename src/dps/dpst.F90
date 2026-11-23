#include "galahad_modules.h"
PROGRAM GALAHAD_DPS_EXAMPLE
   USE GALAHAD_KINDS_precision
   USE GALAHAD_DPS_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_, two = 2.0_rp_
   REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
   REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_, eps = 0.0000000001_rp_
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 99  ! problem dimension
   INTEGER ( KIND = ip_ ) :: i, j, k, nn, sub, pass, var
   TYPE ( SMT_type ) :: H
   REAL ( KIND = rp_ ), DIMENSION( n ) :: C, X
   REAL ( KIND = rp_ ) :: f, f1, f2, delta, sigma, p, tol
   TYPE ( DPS_data_type ) :: data
   TYPE ( DPS_control_type ) :: control
   TYPE ( DPS_inform_type ) :: inform
   TYPE ( SMT_type ) :: H2
   REAL ( KIND = rp_ ), DIMENSION( 3 ) :: C2, X2, X2_star

   H%ne = 132                                  ! set up problem
   f = 0.0_rp_
   CALL SMT_put( H%type, 'COORDINATE', i )
   ALLOCATE( H%row( H%ne ), H%col( H%ne ), H%val( H%ne ), STAT = i )

!  ==============
!  Normal entries
!  ==============

   WRITE( 6, "( /, ' =-=-=-= normal exit tests =-=-=-= ' )" )

! Initialize control parameters

   OPEN( UNIT = 23, STATUS = 'SCRATCH' )
   DO var = 1, 2
     IF ( var == 1 ) THEN
       WRITE( 6, "( /, ' <==== modified-absolute value norm =====>' )" )
     ELSE
       WRITE( 6, "( /, ' <============ Goldfarb norm ============>' )" )
     END IF
     DO sub = 1, 2
       IF ( sub == 1 ) THEN
         delta = one
         WRITE( 6, "( /, ' ===== trust-region subproblem =======', / )" )
       ELSE
         sigma = one
         p = 3.0_rp_
         WRITE( 6, "( /, ' ===== regularization subproblem =====', / )" )
       END IF
       DO pass = 1, 11
         IF ( pass == 1 .OR. pass == 3 )                                       &
           CALL DPS_initialize( data, control, inform )
!        control%symmetric_linear_solver = "ma27 "
         control%symmetric_linear_solver = 'sytr'
!        control%symmetric_linear_solver = 'ma97'
         control%error = 23 ; control%out = 23 ; control%print_level = 10
!        control%error = 6 ; control%out = 6 ; control%print_level = 1
         IF ( var == 1 ) THEN
           control%goldfarb = .FALSE.
         ELSE
           control%goldfarb = .TRUE.
         END IF
         tol = SQRT( EPSILON( one ) )
         IF ( pass == 1 ) THEN
           j = 0 ; k = 0
           DO i = 1, 11
             H%row( j +  1 ) = k + 1 ; H%col( j +  1 ) = k + 1
             H%val( j +  1 ) = two
             H%row( j +  2 ) = k + 2 ; H%col( j +  2 ) = k + 1
             H%val( j +  2 ) = one
             H%row( j +  3 ) = k + 2 ; H%col( j +  3 ) = k + 2
             H%val( j +  3 ) = one
             H%row( j +  4 ) = k + 3 ; H%col( j +  4 ) = k + 3
             H%val( j +  4 ) = zero
             H%row( j +  5 ) = k + 4 ; H%col( j +  5 ) = k + 3
             H%val( j +  5 ) = one
             H%row( j +  6 ) = k + 4 ; H%col( j +  6 ) = k + 4
             H%val( j +  6 ) = zero
             H%row( j +  7 ) = k + 5 ; H%col( j +  7 ) = k + 5
             H%val( j +  7 ) = - one
             H%row( j +  8 ) = k + 5 ; H%col( j +  8 ) = k + 6
             H%val( j +  8 ) = - one
             H%row( j +  9 ) = k + 6 ; H%col( j +  9 ) = k + 6
             H%val( j +  9 ) = - two
             H%row( j + 10 ) = k + 7 ; H%col( j + 10 ) = k + 7
             H%val( j + 10 ) = one
             H%row( j + 11 ) = k + 8 ; H%col( j + 11 ) = k + 8
             H%val( j + 11 ) = eps
             H%row( j + 12 ) = k + 9 ; H%col( j + 12 ) = k + 9
             H%val( j + 12 ) = - ten
             j = j + 12 ; k = k + 9
           END DO
         END IF
         IF ( pass == 2 ) delta = 1000.0_rp_
         IF ( pass == 3 ) THEN
           j = 0 ; k = 0
           DO i = 1, 11
             H%row( j +  1 ) = k + 1 ; H%col( j +  1 ) = k + 1
             H%val( j +  1 ) = two
             H%row( j +  2 ) = k + 2 ; H%col( j +  2 ) = k + 1
             H%val( j +  2 ) = one
             H%row( j +  3 ) = k + 2 ; H%col( j +  3 ) = k + 2
             H%val( j +  3 ) = one
             H%row( j +  4 ) = k + 3 ; H%col( j +  4 ) = k + 3
             H%val( j +  4 ) = two
             H%row( j +  5 ) = k + 4 ; H%col( j +  5 ) = k + 3
             H%val( j +  5 ) = one
             H%row( j +  6 ) = k + 4 ; H%col( j +  6 ) = k + 4
             H%val( j +  6 ) = one
             H%row( j +  7 ) = k + 5 ; H%col( j +  7 ) = k + 5
             H%val( j +  7 ) = two
             H%row( j +  8 ) = k + 5 ; H%col( j +  8 ) = k + 6
             H%val( j +  8 ) = one
             H%row( j +  9 ) = k + 6 ; H%col( j +  9 ) = k + 6
             H%val( j +  9 ) = one
             H%row( j + 10 ) = k + 7 ; H%col( j + 10 ) = k + 7
             H%val( j + 10 ) = one
             H%row( j + 11 ) = k + 8 ; H%col( j + 11 ) = k + 8
             H%val( j + 11 ) = one
             H%row( j + 12 ) = k + 9 ; H%col( j + 12 ) = k + 9
             H%val( j + 12 ) = one
             j = j + 12 ; k = k + 9
           END DO
         END IF
         IF ( pass == 4 ) delta = delta / two
         IF ( pass == 5 ) delta = 0.0001_rp_
         IF ( pass == 6 ) THEN
           j = 0 ; k = 0
           DO i = 1, 11
             H%row( j +  1 ) = k + 1 ; H%col( j +  1 ) = k + 1
             H%val( j +  1 ) = two
             H%row( j +  2 ) = k + 2 ; H%col( j +  2 ) = k + 1
             H%val( j +  2 ) = one
             H%row( j +  3 ) = k + 2 ; H%col( j +  3 ) = k + 2
             H%val( j +  3 ) = one
             H%row( j +  4 ) = k + 3 ; H%col( j +  4 ) = k + 3
             H%val( j +  4 ) = two
             H%row( j +  5 ) = k + 4 ; H%col( j +  5 ) = k + 3
             H%val( j +  5 ) = one
             H%row( j +  6 ) = k + 4 ; H%col( j +  6 ) = k + 4
             H%val( j +  6 ) = one
             H%row( j +  7 ) = k + 5 ; H%col( j +  7 ) = k + 5
             H%val( j +  7 ) = two
             H%row( j +  8 ) = k + 5 ; H%col( j +  8 ) = k + 6
             H%val( j +  8 ) = one
             H%row( j +  9 ) = k + 6 ; H%col( j +  9 ) = k + 6
             H%val( j +  9 ) = one
             H%row( j + 10 ) = k + 7 ; H%col( j + 10 ) = k + 7
             H%val( j + 10 ) = one
             H%row( j + 11 ) = k + 8 ; H%col( j + 11 ) = k + 8
             H%val( j + 11 ) = one
             H%row( j + 12 ) = k + 9 ; H%col( j + 12 ) = k + 9
             H%val( j + 12 ) = zero
             j = j + 12 ; k = k + 9
           END DO
         END IF
         IF ( pass == 7 ) THEN
           delta = 0.1_rp_
           j = 0 ; k = 0
           DO i = 1, 11
             H%row( j +  1 ) = k + 1 ; H%col( j +  1 ) = k + 1
             H%val( j +  1 ) = two
             H%row( j +  2 ) = k + 2 ; H%col( j +  2 ) = k + 1
             H%val( j +  2 ) = one
             H%row( j +  3 ) = k + 2 ; H%col( j +  3 ) = k + 2
             H%val( j +  3 ) = one
             H%row( j +  4 ) = k + 3 ; H%col( j +  4 ) = k + 3
             H%val( j +  4 ) = two
             H%row( j +  5 ) = k + 4 ; H%col( j +  5 ) = k + 3
             H%val( j +  5 ) = one
             H%row( j +  6 ) = k + 4 ; H%col( j +  6 ) = k + 4
             H%val( j +  6 ) = one
             H%row( j +  7 ) = k + 5 ; H%col( j +  7 ) = k + 5
             H%val( j +  7 ) = two
             H%row( j +  8 ) = k + 5 ; H%col( j +  8 ) = k + 6
             H%val( j +  8 ) = one
             H%row( j +  9 ) = k + 6 ; H%col( j +  9 ) = k + 6
             H%val( j +  9 ) = one
             H%row( j + 10 ) = k + 7 ; H%col( j + 10 ) = k + 7
             H%val( j + 10 ) = one
             H%row( j + 11 ) = k + 8 ; H%col( j + 11 ) = k + 8
             H%val( j + 11 ) = one
             H%row( j + 12 ) = k + 9 ; H%col( j + 12 ) = k + 9
             H%val( j + 12 ) = eps
             j = j + 12 ; k = k + 9
           END DO
         END IF
         IF ( pass == 8 ) delta = 100000.0_rp_
         IF ( pass == 9 ) delta = 10.0_rp_
         IF ( pass == 10 ) delta = 10.0_rp_
         IF ( pass == 11 ) delta = 10000.0_rp_

         IF ( pass == 2 ) THEN
           k = 0
           DO i = 1, 11
             C( k + 1 ) = one
             C( k + 2 : k + 9 ) = zero
             k = k + 9
           END DO
         ELSE IF ( pass == 10 ) THEN
           C( : n - 1 ) = 0.000000000001_rp_ ; C( n ) = one
         ELSE
           C = one
         END IF

         IF ( sub == 1 ) THEN
           IF ( pass == 1 .OR. pass == 3 .OR. pass == 6 .OR. pass == 7 ) THEN
             CALL DPS_solve( n, H, C, f, X, data, control, inform,             &
                             delta = delta )
           ELSE IF ( pass == 2 .OR. pass == 10 ) THEN
             CALL DPS_resolve( n, X, data, control, inform,                    &
                               C = C, delta = delta )
           ELSE
             CALL DPS_resolve( n, X, data, control, inform, delta = delta )
           END IF
           f1 = DPS_objective( n, H, C, X, f )
           f2 = 0.5_rp_ * ( DOT_PRODUCT( C, X )                                &
                   - inform%multiplier * delta * delta )
!          IF ( ABS( inform%obj - f1 ) > tol * MAX( one, ABS( inform%obj ) )   &
!               .OR.                                                           &
!               ABS( inform%obj - f2 ) > tol * MAX( one, ABS( inform%obj ) ) ) &
!             WRITE( 6, "( ' new  real f ', 3ES22.14 )" ) inform%obj, f1, f2
         ELSE
           IF ( pass == 1 .OR. pass == 3 .OR. pass == 6 .OR. pass == 7 ) THEN
             CALL DPS_solve( n, H, C, f, X, data, control, inform,             &
                             sigma = sigma, p = p )
           ELSE IF ( pass == 2 .OR. pass == 10 ) THEN
             CALL DPS_resolve( n, X, data, control, inform,                    &
                               C = C, sigma = sigma )
           ELSE
             CALL DPS_resolve( n, X, data, control, inform, sigma = sigma )
           END IF
         END IF

         IF ( pass == 1 .OR. pass == 3 .OR. pass == 6 .OR. pass == 7 ) THEN
           WRITE( 6, "( ' pass ', I3, ' DPS_solve   exit status = ', I6 )" )   &
               pass, inform%status
         ELSE
           WRITE( 6, "( ' pass ', I3, ' DPS_resolve exit status = ', I6 )" )   &
               pass, inform%status
         END IF
       END DO
     END DO
   END DO

!  ===========================================
!  regularization with the powers p = 2 and 4
!  ===========================================

!  H = diag( 1, 2, 3 ) is positive definite, so M = H, and in the variables
!  z = H^1/2 x, ( 1 + lambda ) z = - H^-1/2 c with
!  lambda = sigma ||z||^( p - 2 ). With c = 1, ||H^-1/2 c||^2 = 11/6, so
!  lambda = 1 and x = - H^-1 c / 2 both for p = 2, sigma = 1 and for p = 4,
!  sigma = 24/11

   WRITE( 6, "( /, ' =-=-=-=-= power-two/four tests =-=-=-=-= ', / )" )
   CALL SMT_put( H2%type, 'COORDINATE', i )
   H2%n = 3 ; H2%ne = 3
   ALLOCATE( H2%row( 3 ), H2%col( 3 ), H2%val( 3 ), STAT = i )
   H2%row = (/ 1, 2, 3 /) ; H2%col = (/ 1, 2, 3 /)
   H2%val = (/ one, two, 3.0_rp_ /)
   C2 = one
   X2_star = - (/ one, one / two, one / 3.0_rp_ /) / two
   DO pass = 1, 2
     IF ( pass == 1 ) THEN
       p = two ; sigma = one
     ELSE
       p = 4.0_rp_ ; sigma = 24.0_rp_ / 11.0_rp_
     END IF
     CALL DPS_initialize( data, control, inform )
     control%symmetric_linear_solver = 'sytr'
     CALL DPS_solve( 3_ip_, H2, C2, zero, X2, data, control, inform,          &
                     sigma = sigma, p = p )
     IF ( inform%status /= 0 .OR.                                              &
          MAXVAL( ABS( X2 - X2_star ) ) > SQRT( EPSILON( one ) ) ) THEN
       WRITE( 6, "( ' p = ', F3.1, ': DPS_solve failed, status = ', I0 )" )   &
         p, inform%status
       STOP 1
     END IF
     WRITE( 6, "( ' p = ', F3.1, ': DPS_solve   exit status = ', I6 )" )      &
       p, inform%status
     CALL DPS_terminate( data, control, inform )
   END DO
   DEALLOCATE( H2%row, H2%col, H2%val, H2%type, STAT = i )

!  =============
!  Error entries
!  =============

   WRITE( 6, "( /, ' =-=-=-=-= error exit tests =-=-=-=-= ', / )" )

! Initialize control parameters

   CALL DPS_initialize( data, control, inform )
   DO pass = 1, 4
     delta = one
     control%error = 23 ; control%out = 23 ; control%print_level = 10
     inform%status = 1
     nn = n
     IF ( pass == 1 ) nn = 0
     IF ( pass == 2 ) delta = - one
     IF ( pass == 3 ) CYCLE
     IF ( pass == 4 ) H%ne = - 1

     C = one
     CALL DPS_solve( nn, H, C, f, X, data, control, inform, delta = delta )

     WRITE( 6, "( ' pass ', I3, ' DPS_solve   exit status = ', I6 )" )         &
            pass, inform%status
   END DO
   CLOSE( unit = 23 )

   CALL DPS_terminate( data, control, inform ) !  delete internal workspace
   STOP

CONTAINS

   FUNCTION DPS_objective( n, H, C, X, f )
   REAL ( KIND = rp_ ) :: DPS_objective
   INTEGER ( KIND = ip_ ) :: n
   REAL ( KIND = rp_ ) :: f
   REAL ( KIND = rp_ ), DIMENSION( n ) :: C, X
   TYPE ( SMT_type ) :: H
   REAL ( KIND = rp_ ), DIMENSION( n ) :: HX
   INTEGER ( KIND = ip_ ) :: i, j, k
   HX = 0.0_rp_
   DO k = 1, H%ne
     i = H%row( k ) ; j = H%col( k )
     HX( i ) = HX( i ) + H%val( k ) * X( j )
     IF ( i /= j ) HX( j ) = HX( j ) + H%val( k ) * X( i )
   END DO
   DPS_objective = f + DOT_PRODUCT( C, X ) + 0.5_rp_ * DOT_PRODUCT( X, HX )
   RETURN
   END FUNCTION DPS_objective

END PROGRAM GALAHAD_DPS_EXAMPLE
