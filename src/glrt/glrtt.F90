! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_GLRT_test_deck
   USE GALAHAD_KINDS_precision
   USE GALAHAD_GLRT_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
   REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_, two = 2.0_rp_
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 100 ! problem dimension
   INTEGER ( KIND = ip_ ) :: i, nn, pass
   REAL ( KIND = rp_ ) :: sigma, eps, p
!  REAL ( KIND = rp_ ) :: f
   REAL ( KIND = rp_ ), DIMENSION( n ) :: X, R, VECTOR, H_vector, O
   REAL ( KIND = rp_ ), DIMENSION( 0 ) :: X0, R0, VECTOR0
   TYPE ( GLRT_data_type ) :: data
   TYPE ( GLRT_control_type ) :: control
   TYPE ( GLRT_inform_type ) :: inform

!  ==============
!  Normal entries
!  ==============

   WRITE( 6, "( /, ' ==== normal exits ====== ', / )" )

! Initialize control parameters

   OPEN( UNIT = 23, STATUS = 'SCRATCH' )
!  OPEN( UNIT = 23 )
   DO pass = 1, 14
      IF ( pass /= 4 .AND. pass /= 7 .AND. pass /= 8 )                         &
           CALL GLRT_initialize( data, control, inform )
      control%error = 23 ; control%out = 23 ; control%print_level = 10
      inform%status = 1
      p = 3.0_rp_
      sigma = one ; eps = zero
      IF ( pass == 2 ) control%unitm = .FALSE. ; sigma = 1000.0_rp_
      IF ( pass == 4 ) THEN
           sigma = sigma / two ; inform%status = 6
      END IF
      IF ( pass == 5 ) sigma = 0.0001_rp_
      IF ( pass == 7 ) THEN
         sigma = 0.1_rp_ ; inform%status = 6
      END IF
      IF ( pass == 8 ) THEN
         sigma = 100.0_rp_ ; inform%status = 6
      END IF
      IF ( pass == 9 ) sigma = 10.0_rp_
      IF ( pass == 10 ) sigma = 10.0_rp_
      IF ( pass == 11 ) sigma = 10000.0_rp_
      IF ( pass == 12 .OR. pass == 14 ) eps = one
      IF ( pass == 13 .OR. pass == 14 ) O = one

      IF ( pass == 10 .OR. pass == 11 ) THEN
         R( : n - 1 ) = 0.000000000001_rp_ ; R( n ) = one
      ELSE
         R = one
      END IF
!      IF ( pass == 13 .OR. pass == 14 ) THEN
!       control%error = 6 ; control%out = 6 ; control%print_level = 1
!      END IF

!  Iteration to find the minimizer

      DO
         IF ( pass == 14 ) THEN
           CALL GLRT_solve( n, p, sigma, X, R, VECTOR, data, control,         &
                            inform, eps = eps, O = O )
         ELSE IF ( pass == 13 ) THEN
           CALL GLRT_solve( n, p, sigma, X, R, VECTOR, data, control,         &
                            inform, O = O )
         ELSE IF ( pass == 12 ) THEN
           CALL GLRT_solve( n, p, sigma, X, R, VECTOR, data, control,         &
                            inform, eps = eps )
         ELSE
           CALL GLRT_solve( n, p, sigma, X, R, VECTOR, data, control,         &
                            inform )
         END IF

! Branch as a result of inform%status

         SELECT CASE( inform%status )

!  Form the preconditioned gradient

         CASE( 2 )
            VECTOR = VECTOR / two

!  Form the matrix-vector product

         CASE ( 3 )
            IF ( pass == 2 .OR. pass == 6 .OR. pass == 7 .OR. pass == 8 ) THEN
               H_vector( 1 ) =  two * VECTOR( 1 ) + VECTOR( 2 )
               DO i = 2, n - 1
                 H_vector( i ) = VECTOR( i - 1 ) + two * VECTOR( i ) +         &
                                 VECTOR( i + 1 )
               END DO
               H_vector( n ) = VECTOR( n - 1 ) + two * VECTOR( n )
            ELSE IF ( pass == 9 ) THEN
               H_vector( 1 ) = VECTOR( 1 ) + VECTOR( 2 )
               DO i = 2, n - 1
                 H_vector( i ) = VECTOR( i - 1 ) - two * VECTOR( i ) +         &
                                 VECTOR( i + 1 )
               END DO
               H_vector( n ) = VECTOR( n - 1 ) + VECTOR( n )
            ELSE IF ( pass == 10 .OR. pass == 11 ) THEN
              H_vector( 1 ) = - two * VECTOR( 1 )
              H_vector( 2 : n - 1 ) = 0.0001_rp_ * VECTOR( 2 : n - 1 )
              H_vector( n ) = - VECTOR( n )
            ELSE
               H_vector( 1 ) = - two * VECTOR( 1 ) + VECTOR( 2 )
               DO i = 2, n - 1
                 H_vector( i ) = VECTOR( i - 1 ) - two * VECTOR( i ) +         &
                                 VECTOR( i + 1 )
               END DO
               H_vector( n ) = VECTOR( n - 1 ) - two * VECTOR( n )
            END IF
            VECTOR = H_vector

!  Restart

         CASE ( 4 )
            IF ( pass == 10 .OR. pass == 11 ) THEN
               R( : n - 1 ) = 0.000000000001_rp_
               R( n ) = one
            ELSE
               R = one
            END IF

!  Form the product with the preconditioner

         CASE( 5 )
            VECTOR = two * VECTOR

!  Successful return

         CASE ( - 2 : 0 )
            EXIT

!  Error returns

         CASE DEFAULT
            EXIT
         END SELECT
      END DO

      WRITE( 6, "( ' pass ', I3, ' GLRT_solve exit status = ', I6 )" )         &
             pass, inform%status
!     WRITE( 6, "( ' its, solution and Lagrange multiplier = ', I6, 2ES12.4 )")&
!               inform%iter + inform%iter_pass2, inform%obj_regularized, &
!               inform%multiplier
      IF ( pass /= 3 .AND. pass /= 6 .AND. pass /= 7 )                         &
        CALL GLRT_terminate( data, control, inform ) ! delete internal workspace
   END DO

!  =============
!  Error entries
!  =============

   WRITE( 6, "( /, ' ==== error exits ====== ', / )" )

! Initialize control parameters

   DO pass = 2, 9
      sigma = one
      eps = zero
      p = 3.0_rp_
      CALL GLRT_initialize( data, control, inform )
      control%error = 23 ; control%out = 23 ; control%print_level = 10
      inform%status = 1
      nn = n
      IF ( pass == 2 ) control%itmax = 0
      IF ( pass == 3 ) inform%status = 0
      IF ( pass == 4 ) nn = 0
      IF ( pass == 5 ) sigma = - one
      IF ( pass == 6 ) CYCLE
      IF ( pass == 7 ) eps = - one
      IF ( pass == 8 ) p = one
      IF ( pass == 9 ) p = two
      IF ( pass == 9 ) control%unitm = .FALSE.

      R = one

!  Iteration to find the minimizer

      DO
         IF ( pass == 4 ) THEN
           CALL GLRT_solve( nn, p, sigma, X0, R0,                          &
                            VECTOR0, data, control, inform )
         ELSE IF ( pass == 7 ) THEN
           CALL GLRT_solve( nn, p, sigma, X( : nn ), R( : nn ),            &
                            VECTOR( : nn ), data, control, inform, eps = eps )
         ELSE
           CALL GLRT_solve( nn, p, sigma, X( : nn ), R( : nn ),            &
                            VECTOR( : nn ), data, control, inform )
         END IF

! Branch as a result of inform%status

         SELECT CASE( inform%status )

!  Form the preconditioned gradient

         CASE( 2 )
            IF ( pass /= 3 ) THEN
               VECTOR = VECTOR / two
            ELSE
               VECTOR = - VECTOR / two
            END IF

!  Form the matrix-vector product

         CASE ( 3 )
            H_vector( 1 ) = - two * VECTOR( 1 ) + VECTOR( 2 )
            DO i = 2, n - 1
              H_vector( i ) = VECTOR( i - 1 ) - two * VECTOR( i ) +         &
                              VECTOR( i + 1 )
            END DO
            H_vector( n ) = VECTOR( n - 1 ) - two * VECTOR( n )
            VECTOR = H_vector

!  Restart

         CASE ( 4 )
            R = one

!  Form the product with the preconditioner

         CASE( 5 )
            IF ( pass /= 3 ) THEN
               VECTOR = two * VECTOR
            ELSE
               VECTOR = - two * VECTOR
            END IF

!  Successful return

         CASE ( - 2 : 0 )
            EXIT

!  Error returns

         CASE DEFAULT
            EXIT
         END SELECT
      END DO

      WRITE( 6, "( ' pass ', I3, ' GLRT_solve exit status = ', I6 )" )         &
             pass, inform%status
      CALL GLRT_terminate( data, control, inform ) !  delete internal workspace
   END DO
   CLOSE( unit = 23 )
   WRITE( 6, "( /, ' tests completed' )" )

!  STOP
   END PROGRAM GALAHAD_GLRT_test_deck
