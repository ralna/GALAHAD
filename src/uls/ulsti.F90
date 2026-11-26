! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_ULS_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SYMBOLS
   USE GALAHAD_ULS_precision
   IMPLICIT NONE
   TYPE ( SMT_type ) :: matrix
   TYPE ( ULS_full_data_type ) :: data
   TYPE ( ULS_control_type ) control
   TYPE ( ULS_inform_type ) :: inform
   INTEGER ( KIND = ip_ ), DIMENSION( 0 ) :: null
   INTEGER ( KIND = ip_ ) :: i, storage_type, status, failures
   INTEGER ( KIND = ip_ ), PARAMETER :: m = 5, n = 5, ne  = 7
   REAL ( KIND = rp_ ) :: B( n ), X( n )
   INTEGER ( KIND = ip_ ), DIMENSION( ne ) :: row = (/ 1, 2, 2, 3, 3, 4, 5 /)
   INTEGER ( KIND = ip_ ), DIMENSION( ne ) :: col = (/ 1, 1, 5, 2, 3, 3, 4 /)
   INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ) :: ptr = (/ 1, 2, 4, 6, 7, 8 /)
   REAL ( KIND = rp_ ), DIMENSION( ne ) ::                                     &
     val = (/ 2.0_rp_, 3.0_rp_, 6.0_rp_, 4.0_rp_,  1.0_rp_, 5.0_rp_, 1.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( n * n ) ::                                  &
     dense = (/ 2.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,                   &
                3.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 6.0_rp_,                   &
                0.0_rp_, 4.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_,                   &
                0.0_rp_, 0.0_rp_, 5.0_rp_, 0.0_rp_, 0.0_rp_,                   &
                0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_, 0.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( n ) ::                                      &
     rhs = (/ 2.0_rp_,  33.0_rp_,  11.0_rp_,  15.0_rp_,  4.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( n ) ::                                      &
     rhst = (/ 8.0_rp_,  12.0_rp_,  23.0_rp_,  5.0_rp_,  12.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( n ) ::                                      &
     SOL = (/ 1.0_rp_,  2.0_rp_,  3.0_rp_,  4.0_rp_,  5.0_rp_ /)
!  the 3 x 2 matrix A_r = ( 1 0 ; 0 2 ; 1 1 ), and its 2 x 3 transpose
   INTEGER ( KIND = ip_ ), PARAMETER :: m_r = 3, n_r = 2, ne_r = 4
   INTEGER ( KIND = ip_ ), DIMENSION( ne_r ) :: row_r = (/ 1, 2, 3, 3 /)
   INTEGER ( KIND = ip_ ), DIMENSION( ne_r ) :: col_r = (/ 1, 2, 1, 2 /)
   INTEGER ( KIND = ip_ ), DIMENSION( m_r + 1 ) :: ptr_r = (/ 1, 2, 3, 5 /)
   REAL ( KIND = rp_ ), DIMENSION( ne_r ) ::                                   &
     val_r = (/ 1.0_rp_, 2.0_rp_, 1.0_rp_, 1.0_rp_ /)
   INTEGER ( KIND = ip_ ), DIMENSION( ne_r ) :: col_t = (/ 1, 3, 2, 3 /)
   INTEGER ( KIND = ip_ ), DIMENSION( n_r + 1 ) :: ptr_t = (/ 1, 3, 5 /)
   REAL ( KIND = rp_ ), DIMENSION( ne_r ) ::                                   &
     val_t = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 1.0_rp_ /)
   REAL ( KIND = rp_ ) :: X_r( m_r )

   WRITE( 6,"( ' storage         RHS   refine   RHST  refine')")
! read matrix order and number of entries
!  DO storage_type = 1, 3
   DO storage_type = 3, 3
! initialize the structures for the gls solver
     CALL ULS_initialize( 'gls ', data, control, inform )
     IF ( inform%status < 0 ) THEN
       CALL ULS_information( data, inform, status )
       WRITE( 6, "( '  fail in initialize, status = ', i0 )", advance = 'no' ) &
         inform%status
       WRITE( 6, "( '' )" )
       CYCLE
     END IF
! factorize! analyse the matrix structure
     SELECT CASE( storage_type )
     CASE ( 1 )
       WRITE( 6, "( A15 )", advance = 'no' ) " coordinate    "
       CALL ULS_factorize_matrix( control, data, status, m, n,                 &
                                  'coordinate', ne, val, row, col, null )
     CASE ( 2 )
       WRITE( 6, "( A15 )", advance = 'no' ) " sparse by rows"
       CALL ULS_factorize_matrix( control, data, status, m, n,                 &
                                  'sparse_by_rows', ne, val, null, col, ptr )
     CASE ( 3 )
       WRITE( 6, "( A15 )", advance = 'no' ) " dense         "
       CALL ULS_factorize_matrix( control, data, status, m, n,                 &
                                  'dense', ne, dense, null, null, null )
     END SELECT
     IF ( status < 0 ) THEN
       CALL ULS_information( data, inform, status )
       WRITE( 6, "( '  fail in analyse, status = ', I0 )", advance = 'no' )    &
         inform%status
       WRITE( 6, "( '' )" )
       CYCLE
     END IF
! solve without refinement
     X = rhs
     CALL ULS_solve_system( data, status, X, .FALSE. )
     IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                            &
             <= EPSILON( 1.0_rp_ ) ** 0.5 ) THEN
       WRITE( 6, "( '   ok  ' )", advance = 'no' )
     ELSE
       WRITE( 6, "( '  fail ' )", advance = 'no' )
     END IF
! perform one refinement
     control%max_iterative_refinements = 1
     CALL ULS_reset_control( control, data, status )
     X = rhs
     CALL ULS_solve_system( data, status, X, .FALSE. )
     IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                            &
             <= EPSILON( 1.0_rp_ ) ** 0.5 ) THEN
       WRITE( 6, "( '    ok  ' )", advance = 'no' )
     ELSE
       WRITE( 6, "( '  fail  ' )", advance = 'no' )
     END IF
! solve transpose system without refinement
     X = rhst
     CALL ULS_solve_system( data, status, X, .TRUE. )
     IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                            &
             <= EPSILON( 1.0_rp_ ) ** 0.5 ) THEN
       WRITE( 6, "( '   ok  ' )", advance = 'no' )
     ELSE
       WRITE( 6, "( '  fail ' )", advance = 'no' )
     END IF
! perform one refinement
     control%max_iterative_refinements = 1
     CALL ULS_reset_control( control, data, status )
     X = rhst
     CALL ULS_solve_system( data, status, X, .TRUE. )
     IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                            &
             <= EPSILON( 1.0_rp_ ) ** 0.5 ) THEN
       WRITE( 6, "( '    ok  ' )" )
     ELSE
       WRITE( 6, "( '  fail  ' )" )
     END IF
     CALL ULS_terminate( data, control, inform )
   END DO

!  =====================================================================
!  error statuses from the factorization, and rectangular matrices with
!  the LAPACK solver getr, which is always available
!  =====================================================================

   failures = 0

!  an error found by the factorization (m < 1) is reported

   CALL ULS_initialize( 'getr', data, control, inform )
   CALL ULS_factorize_matrix( control, data, status, 0_ip_, n_r,             &
                              'coordinate', ne_r, val_r, row_r, col_r, null )
   IF ( status /= GALAHAD_error_restrictions ) THEN
     WRITE( 6, "( ' m = 0: status ', I0, ' /= ', I0 )" )                       &
       status, GALAHAD_error_restrictions
     failures = failures + 1
   END IF
   CALL ULS_terminate( data, control, inform )

!  getr factorizes rectangular matrices, given by co-ordinates or by rows,
!  and reports their rank, but cannot solve systems with them

   DO storage_type = 1, 3
     CALL ULS_initialize( 'getr', data, control, inform )
     SELECT CASE( storage_type )
     CASE ( 1 )
       CALL ULS_factorize_matrix( control, data, status, m_r, n_r,           &
                                  'coordinate', ne_r, val_r, row_r, col_r,   &
                                  null )
     CASE ( 2 )
       CALL ULS_factorize_matrix( control, data, status, m_r, n_r,           &
                                  'sparse_by_rows', ne_r, val_r, null,       &
                                  col_r, ptr_r )
     CASE ( 3 )
       CALL ULS_factorize_matrix( control, data, status, n_r, m_r,           &
                                  'sparse_by_rows', ne_r, val_t, null,       &
                                  col_t, ptr_t )
     END SELECT
     IF ( status /= GALAHAD_ok ) THEN
       WRITE( 6, "( ' rectangular ', I0, ': factorize status ', I0 )" )        &
         storage_type, status
       failures = failures + 1
     ELSE
       CALL ULS_information( data, inform, status )
       IF ( inform%rank /= n_r ) THEN
         WRITE( 6, "( ' rectangular ', I0, ': rank ', I0, ' /= ', I0 )" )      &
           storage_type, inform%rank, n_r
         failures = failures + 1
       END IF
       X_r = 1.0_rp_
       CALL ULS_solve_system( data, status, X_r, .FALSE. )
       IF ( status /= GALAHAD_unavailable_option ) THEN
         WRITE( 6, "( ' rectangular ', I0, ': solve status ', I0, ' /= ',      &
        &  I0 )" ) storage_type, status, GALAHAD_unavailable_option
         failures = failures + 1
       END IF
     END IF
     CALL ULS_terminate( data, control, inform )
   END DO

   IF ( failures > 0 ) THEN
     WRITE( 6, "( 1X, I0, ' getr check(s) failed' )" ) failures
     STOP 1
   END IF
   WRITE( 6, "( ' getr checks ok' )" )
   STOP
   END PROGRAM GALAHAD_ULS_interface_test
