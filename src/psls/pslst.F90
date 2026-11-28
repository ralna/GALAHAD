! THIS VERSION: GALAHAD 5.6 - 2026-11-28 AT 12:42 GMT.
#include "galahad_modules.h"
   PROGRAM PSLS_TEST_PROGRAM
   USE GALAHAD_KINDS_precision
   USE GALAHAD_PSLS_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   TYPE ( SMT_type ) :: matrix
   TYPE ( PSLS_data_type ) :: data
   TYPE ( PSLS_control_type ) control
   TYPE ( PSLS_inform_type ) :: inform
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 5
   INTEGER ( KIND = ip_ ), PARAMETER :: ne = 7
   REAL ( KIND = rp_ ) :: X( n )
   INTEGER ( KIND = ip_ ) :: SUB( 3 ) = (/ 1, 3, 5 /)
   INTEGER ( KIND = ip_ ) :: i, s, stat_faf, stat_sol, prec
   CHARACTER ( LEN = 1 ) :: st
   TYPE ( SMT_type ) :: A3
   REAL ( KIND = rp_ ) :: X3( 3 ), P_inv( 3, 3 ), minor
   LOGICAL :: skip

!  -------------------------------------------------------------------
!  the Schnabel-Eskow preconditioner must be positive definite, even
!  for an indefinite matrix and a solver that does not modify its
!  pivots (GMPS is then used instead). For the indefinite
!  A = ( 1 0 0 ; 0 -2 1 ; 0 1 3 ), check that P^{-1} has positive leading
!  principal minors
!  -------------------------------------------------------------------

   WRITE( 6, "( ' Schnabel-Eskow test' )" )
   CALL SMT_put( A3%type, 'COORDINATE', s )
   A3%n = 3 ; A3%ne = 4
   ALLOCATE( A3%val( 4 ), A3%row( 4 ), A3%col( 4 ) )
   A3%row = (/ 1, 2, 3, 3 /) ; A3%col = (/ 1, 2, 2, 3 /)
   A3%val = (/ 1.0_rp_, - 2.0_rp_, 1.0_rp_, 3.0_rp_ /)
   CALL PSLS_initialize( data, control, inform )
   CALL WHICH_sls( control )
   control%preconditioner = 4
   CALL PSLS_form_and_factorize( A3, data, control, inform )

!  skip the test if the solver cannot provide the GMPS modification, or
!  cannot tell whether P is positive definite

   skip = inform%status == GALAHAD_unavailable_option .OR.                     &
          inform%status == GALAHAD_error_unknown_solver .OR.                   &
          ( inform%status == GALAHAD_ok .AND. inform%preconditioner == 4 .AND. &
            inform%SLS_inform%negative_eigenvalues < 0 )
   IF ( skip ) THEN
     WRITE( 6, "( ' skipped, status = ', I0 )" ) inform%status
   ELSE IF ( inform%status /= GALAHAD_ok ) THEN
     WRITE( 6, "( ' form status = ', I0, ' FAILED' )" ) inform%status
     STOP 1
   ELSE
     DO i = 1, 3
       X3 = 0.0_rp_ ; X3( i ) = 1.0_rp_
       CALL PSLS_apply( X3, data, control, inform )
       P_inv( : , i ) = X3
     END DO
     minor = P_inv( 1, 1 ) * P_inv( 2, 2 ) - P_inv( 1, 2 ) * P_inv( 2, 1 )
     IF ( P_inv( 1, 1 ) <= 0.0_rp_ .OR. minor <= 0.0_rp_ .OR.                  &
          P_inv( 1, 1 ) * ( P_inv( 2, 2 ) * P_inv( 3, 3 )                      &
            - P_inv( 2, 3 ) * P_inv( 3, 2 ) )                                  &
          - P_inv( 1, 2 ) * ( P_inv( 2, 1 ) * P_inv( 3, 3 )                    &
            - P_inv( 2, 3 ) * P_inv( 3, 1 ) )                                  &
          + P_inv( 1, 3 ) * ( P_inv( 2, 1 ) * P_inv( 3, 2 )                    &
            - P_inv( 2, 2 ) * P_inv( 3, 1 ) ) <= 0.0_rp_ ) THEN
       WRITE( 6, "( ' P is not positive definite FAILED' )" )
       STOP 1
     END IF
     WRITE( 6, "( ' P positive definite', / )" )
   END IF
   CALL PSLS_terminate( data, control, inform )
   DEALLOCATE( A3%val, A3%row, A3%col, A3%type )
!  ---------------------------------------------
!  test basic storage and preconditioner options
!  ---------------------------------------------
   WRITE( 6, "( ' storage and preconditioner tests', / )" )
! run through storage types
   DO i = 1, 3
! allocate and set lower triangle of matrix
    IF ( i == 1 ) THEN ! co-ordinate form
       st = 'C'
       CALL SMT_put( matrix%type, 'COORDINATE', s )
       matrix%n = n ; matrix%ne = ne
       ALLOCATE( matrix%val( ne ), matrix%row( ne ), matrix%col( ne ) )
       matrix%row = (/ 1, 2, 3, 3, 4, 5, 5 /)
       matrix%col = (/ 1, 1, 2, 3, 3, 2, 5 /)
       matrix%val = (/ 2.0_rp_, 3.0_rp_, 4.0_rp_, 1.0_rp_, 5.0_rp_,            &
                               6.0_rp_, 1.0_rp_ /)
     ELSE IF ( i == 2 ) THEN ! sparse-row form
       st = 'R'
       CALL SMT_put( matrix%type, 'SPARSE_BY_ROWS', s )
       matrix%n = n
       ALLOCATE( matrix%val( ne ), matrix%col( ne ), matrix%ptr( n + 1 ) )
       matrix%ptr = (/ 1, 2, 3, 5, 6, 8 /)
       matrix%col = (/ 1, 1, 2, 3, 3, 2, 5 /)
       matrix%val = (/ 2.0_rp_, 3.0_rp_, 4.0_rp_, 1.0_rp_, 5.0_rp_,            &
                               6.0_rp_, 1.0_rp_ /)
     ELSE IF ( i == 3 ) THEN ! dense form
       st = 'D'
       CALL SMT_put( matrix%type, 'DENSE', s )
       matrix%n = n
       ALLOCATE( matrix%val( n * ( n + 1 ) / 2 ) )
       matrix%val = (/ 2.0_rp_, 3.0_rp_, 0.0_rp_, 0.0_rp_, 4.0_rp_, 1.0_rp_,   &
                       0.0_rp_, 0.0_rp_, 5.0_rp_, 0.0_rp_, 0.0_rp_, 6.0_rp_,   &
                       0.0_rp_, 0.0_rp_, 1.0_rp_ /)
     END IF
! run through preconditioners
     DO prec = - 1, 7
       IF ( prec == 3 .OR. prec == 6 ) CYCLE
! specify the solver used by SLS (in this case sils)
       CALL PSLS_initialize( data, control, inform )
       CALL WHICH_sls( control )
       control%preconditioner = prec
       control%semi_bandwidth = 1  ! semi-bandwidth of one
!      control%definite_linear_solver = 'sils'
! problem setup complete. form and factorize the preconditioner, P
       CALL PSLS_form_and_factorize( matrix, data, control, inform )
       stat_faf = inform%status
       IF ( stat_faf == 0 ) THEN
! use the factors to solve P x = b, with b input in x
         X( : n ) = (/ 8.0_rp_,  45.0_rp_,  31.0_rp_,  15.0_rp_,  17.0_rp_ /)
         CALL PSLS_apply( X, data, control, inform )
         stat_sol = inform%status
!        IF ( inform%status == 0 ) THEN
!          WRITE( 6, "( ' PSLS - Preconditioned solution is ', 5F6.2 )" ) X
!        ELSE
!          WRITE( 6, "( ' PSLS - exit status = ', I0 )" ) inform%status
!        END IF
       ELSE
         stat_sol = - 1
       END IF
       WRITE( 6, "( I2, A, ' prec:storage: status form & fact = ', I3,         &
      &           ' solve = ', I3 )" ) prec, st, stat_faf, stat_sol
! clean up
       CALL PSLS_terminate( data, control, inform )
!      GO TO 9
     END DO
     IF ( i == 1 ) THEN ! co-ordinate form
       DEALLOCATE( matrix%type, matrix%val, matrix%row, matrix%col )
     ELSE IF ( i == 2 ) THEN ! spares-row form
       DEALLOCATE( matrix%type, matrix%val, matrix%col, matrix%ptr )
     ELSE IF ( i == 3 ) THEN ! dense form
       DEALLOCATE( matrix%type, matrix%val )
     END IF
   END DO
!  -------------------------------------------------------------
!  test basic storage and preconditioner options for submatrices
!  -------------------------------------------------------------
   WRITE( 6, "( /, ' storage and preconditioner tests with submatrices', / )" )
! run through storage types
   DO i = 1, 3
! allocate and set lower triangle of matrix
    IF ( i == 1 ) THEN ! co-ordinate form
       st = 'C'
       CALL SMT_put( matrix%type, 'COORDINATE', s )
       matrix%n = n ; matrix%ne = ne
       ALLOCATE( matrix%val( ne ), matrix%row( ne ), matrix%col( ne ) )
       matrix%row = (/ 1, 2, 3, 3, 4, 5, 5 /)
       matrix%col = (/ 1, 1, 2, 3, 3, 2, 5 /)
       matrix%val = (/ 2.0_rp_, 3.0_rp_, 4.0_rp_, 1.0_rp_, 5.0_rp_,            &
                       6.0_rp_, 1.0_rp_ /)
     ELSE IF ( i == 2 ) THEN ! sparse-row form
       st = 'R'
       CALL SMT_put( matrix%type, 'SPARSE_BY_ROWS', s )
       matrix%n = n
       ALLOCATE( matrix%val( ne ), matrix%col( ne ), matrix%ptr( n + 1 ) )
       matrix%ptr = (/ 1, 2, 3, 5, 6, 8 /)
       matrix%col = (/ 1, 1, 2, 3, 3, 2, 5 /)
       matrix%val = (/ 2.0_rp_, 3.0_rp_, 4.0_rp_, 1.0_rp_, 5.0_rp_,            &
                       6.0_rp_, 1.0_rp_ /)
     ELSE IF ( i == 3 ) THEN ! dense form
       st = 'D'
       CALL SMT_put( matrix%type, 'DENSE', s )
       matrix%n = n
       ALLOCATE( matrix%val( n * ( n + 1 ) / 2 ) )
       matrix%val = (/ 2.0_rp_, 3.0_rp_, 0.0_rp_, 0.0_rp_, 4.0_rp_, 1.0_rp_,   &
                       0.0_rp_, 0.0_rp_, 5.0_rp_, 0.0_rp_, 0.0_rp_, 6.0_rp_,   &
                       0.0_rp_, 0.0_rp_, 1.0_rp_ /)
     END IF
! run through preconditioners
     DO prec = - 1, 7
       IF ( prec == 3 .OR. prec == 6 ) CYCLE
! specify the solver used by SLS (in this case sils)
       CALL PSLS_initialize( data, control, inform )
       CALL WHICH_sls( control )
       control%preconditioner = prec
       control%semi_bandwidth = 1  ! semi-bandwidth of one
!      control%definite_linear_solver = 'sils'
! problem setup complete. form and factorize the preconditioner, P
       CALL PSLS_form_and_factorize( matrix, data, control, inform, SUB = SUB )
       stat_faf = inform%status
       IF ( stat_faf == 0 ) THEN
! use the factors to solve P x = b, with b input in x
         X( : n ) = (/ 8.0_rp_,  45.0_rp_,  31.0_rp_,  15.0_rp_,  17.0_rp_ /)
         CALL PSLS_apply( X, data, control, inform )
         stat_sol = inform%status
!        IF ( inform%status == 0 ) THEN
!          WRITE( 6, "( ' PSLS - Preconditioned solution is ', 5F6.2 )" ) X
!        ELSE
!          WRITE( 6, "( ' PSLS - exit status = ', I0 )" ) inform%status
!        END IF
       ELSE
         stat_sol = - 1
       END IF
       WRITE( 6, "( I2, A, ' prec:storage: status form & fact = ', I3,         &
      &           ' solve = ', I3 )" ) prec, st, stat_faf, stat_sol
! clean up
       CALL PSLS_terminate( data, control, inform )
     END DO
     IF ( i == 1 ) THEN ! co-ordinate form
       DEALLOCATE( matrix%type, matrix%val, matrix%row, matrix%col )
     ELSE IF ( i == 2 ) THEN ! spares-row form
       DEALLOCATE( matrix%type, matrix%val, matrix%col, matrix%ptr )
     ELSE IF ( i == 3 ) THEN ! dense form
       DEALLOCATE( matrix%type, matrix%val )
     END IF
   END DO
!  -----------------
! Test error returns
!  -----------------
   WRITE( 6, "( /, ' error tests', / )" )
! specify the generic problem
   CALL SMT_put( matrix%type, 'COORDINATE', s )
   matrix%n = n ; matrix%ne = ne
   ALLOCATE( matrix%val( ne ), matrix%row( ne ), matrix%col( ne ) )
   matrix%row = (/ 1, 2, 3, 3, 4, 5, 5 /)
   matrix%col = (/ 1, 1, 2, 3, 3, 2, 5 /)
   matrix%val = (/ 2.0_rp_, 3.0_rp_, 4.0_rp_, 1.0_rp_, 5.0_rp_, 6.0_rp_,       &
                   1.0_rp_ /)
   DO i = 1, 4
! specify the solver used by SLS (in this case sils)
     CALL PSLS_initialize( data, control, inform )
     CALL WHICH_sls( control )
     control%preconditioner = 2  ! band preconditioner
     control%semi_bandwidth = 1  ! semi-bandwidth of one
!    control%definite_linear_solver = 'sils'
! provoke error condition
     SELECT CASE ( i )
     CASE( 1 )
       matrix%n = - 1 ; matrix%ne = ne
     CASE( 2 )
       CALL SMT_put( matrix%type, 'BAD', s )
       matrix%n = n ; matrix%ne = ne
     CASE( 3 )
       CALL SMT_put( matrix%type, 'COORDINATE', s )
       control%preconditioner = 5
       control%definite_linear_solver = 'bad'
     CASE( 4 )
!      control%definite_linear_solver = 'sils'
       control%preconditioner = 10
     CASE( 5 )
       control%preconditioner = 7
       control%mi28_lsize = - 1
       control%mi28_rsize = - 1
     CASE DEFAULT
       control%preconditioner = 2
     END SELECT
! problem setup complete. form and factorize the preconditioner, P
     CALL PSLS_form_and_factorize( matrix, data, control, inform )
     stat_faf = inform%status
     WRITE( 6, "( ' status form  = ', I3 )" ) stat_faf
     CALL PSLS_terminate( data, control, inform )
   END DO
!9 CONTINUE

!  ----------------------------------------------------------------
!  check that applying a preconditioner that could not be formed is
!  reported as an error (preconditioner 9 does not exist)
!  ----------------------------------------------------------------

   WRITE( 6, "( /, ' apply after failure test' )" )
   CALL PSLS_initialize( data, control, inform )
   CALL WHICH_sls( control )
   control%preconditioner = 9
   CALL PSLS_form_and_factorize( matrix, data, control, inform )
   stat_faf = inform%status
   X = 1.0_rp_
   CALL PSLS_apply( X, data, control, inform )
   stat_sol = inform%status
   WRITE( 6, "( ' status form  = ', I3, ' status apply = ', I3 )" )           &
     stat_faf, stat_sol
   IF ( stat_faf == GALAHAD_ok .OR.                                            &
        stat_sol /= GALAHAD_error_call_order ) THEN
     WRITE( 6, "( ' apply after failure test FAILED' )" )
     STOP 1
   END IF
   CALL PSLS_terminate( data, control, inform )
   DEALLOCATE( matrix%type, matrix%val, matrix%row, matrix%col )
   STOP

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( PSLS_control_type ) :: control
#include "galahad_sls_defaults_ls.h"
     control%symmetric_linear_solver = symmetric_linear_solver
     control%definite_linear_solver = definite_linear_solver
!    control%definite_linear_solver = 'sytr'
     END SUBROUTINE WHICH_sls

   END PROGRAM PSLS_TEST_PROGRAM
