! THIS VERSION: GALAHAD 5.6.1 - 2026-11-02
!
! Edge-case tests for the public SLBLT interface (GALAHAD_SLBLT_precision).
! Covers:
!   * invalid / extreme input arguments (n < 0, n = 0, n = 1, bad ptr,
!     out-of-range and duplicate entries, bad orders, wrong call sequences,
!     bad job / nrhs / ldx in solve, posdef/indef enquiry mismatches);
!   * extreme matrix values (entries scaled near HUGE**0.25 and TINY**0.25,
!     exactly singular matrices, zero diagonals needing 2x2 pivots);
!   * control parameters at non-default values (ordering, nemin, scaling,
!     pivot_method, failed_pivot_method, u, small, action, block_size,
!     small_subtree_threshold, ignore_numa, multiplier, print_level and units).
! Each check prints "ok" or "FAIL"; the program stops with exit code 1 if any
! check fails.

#include "galahad_modules.h"

   PROGRAM GALAHAD_SLBLT_edge_test_program
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SLBLT_precision
   USE GALAHAD_SLBLT_types_precision, ONLY :                                   &
     SLBLT_SUCCESS, SLBLT_ERROR_CALL_SEQUENCE, SLBLT_ERROR_A_N_OOR,            &
     SLBLT_ERROR_A_PTR, SLBLT_ERROR_A_ALL_OOR, SLBLT_ERROR_SINGULAR,           &
     SLBLT_ERROR_NOT_POS_DEF, SLBLT_ERROR_PTR_ROW, SLBLT_ERROR_ORDER,          &
     SLBLT_ERROR_VAL, SLBLT_ERROR_X_SIZE, SLBLT_ERROR_JOB_OOR,                 &
     SLBLT_ERROR_NOT_LLT, SLBLT_ERROR_NOT_LDLT, SLBLT_ERROR_NO_SAVED_SCALING,  &
     SLBLT_ERROR_NO_METIS, SLBLT_WARNING_IDX_OOR, SLBLT_WARNING_DUP_IDX,       &
     SLBLT_WARNING_DUP_AND_OOR, SLBLT_WARNING_FACT_SINGULAR,                   &
     SLBLT_WARNING_MATCH_ORD_NO_SCALE
   IMPLICIT NONE

   REAL( rp_ ), PARAMETER :: zero = 0.0_rp_, one = 1.0_rp_
   REAL( rp_ ), PARAMETER :: eps = EPSILON( one )
   REAL( rp_ ), PARAMETER :: tol = 1000.0_rp_ * eps       ! backward-error tol
   REAL( rp_ ), PARAMETER :: loose = 100.0_rp_ * SQRT( eps )
   INTEGER( ip_ ) :: nfail, npass

   nfail = 0 ; npass = 0
#ifdef REAL_32
   WRITE( 6, "( '  slblt edge-case tests, single precision', / )" )
#elif REAL_128
   WRITE( 6, "( '  slblt edge-case tests, quadruple precision', / )" )
#else
   WRITE( 6, "( '  slblt edge-case tests, double precision', / )" )
#endif

   WRITE( 6, "( ' -- invalid arguments and call sequence --' )" )
   CALL case_n_negative( )
   CALL case_n_zero( )
   CALL case_bad_ordering_value( )
   CALL case_bad_user_order( )
   CALL case_ordering2_no_val( )
   CALL case_bad_ptr( )
   CALL case_oor_and_dup( )
   CALL case_all_oor( )
   CALL case_call_sequence( )
   CALL case_nocheck_needs_ptr_row( )
   CALL case_solve_args( )
   CALL case_posdef_partial_solves( )
   CALL case_enquire_mismatch( )
   CALL case_no_saved_scaling( )

   WRITE( 6, "( /, ' -- extreme matrices --' )" )
   CALL case_n_one( )
   CALL case_singular( )
   CALL case_zero_diagonal_2x2( )
   CALL case_extreme_scale( )
   CALL case_not_posdef( )
   CALL case_large_n( )
   CALL case_coord_input( )
   CALL case_ptr32( )

   WRITE( 6, "( /, ' -- non-default control parameters --' )" )
   CALL case_orderings( )
   CALL case_nemin( )
   CALL case_scalings( )
   CALL case_pivot_methods( )
   CALL case_threshold_u( )
   CALL case_tpp_emergency_4x4( )
   CALL case_small( )
   CALL case_blocking_and_threads( )
   CALL case_printing( )
   CALL case_refactor_reuse( )
!  subtree splitting, including delayed pivots that cross subtree boundaries
!  (failures here used to crash SLBLT_free, so they are kept last)
   CALL case_subtree_delay_3x3( )
   CALL case_subtree_partition( )

   WRITE( 6, "( /, '  ', I0, ' checks passed, ', I0, ' failed' )" )         &
     npass, nfail
   IF ( nfail /= 0 ) THEN
     WRITE( 6, "( '  slblt edge-case tests FAILED' )" )
     STOP 1
   END IF
   WRITE( 6, "( '  slblt edge-case tests completed' )" )

 CONTAINS

!  =========================================================================
!  helpers
!  =========================================================================

   SUBROUTINE report( cond, label )
   LOGICAL, INTENT( IN ) :: cond
   CHARACTER( * ), INTENT( IN ) :: label
   IF ( cond ) THEN
     WRITE( 6, "( '  ok   : ', A )" ) label
     npass = npass + 1
   ELSE
     WRITE( 6, "( '  FAIL : ', A )" ) label
     nfail = nfail + 1
   END IF
   FLUSH( 6 )
   END SUBROUTINE report

   SUBROUTINE report_flag( got, expected, label )
   INTEGER( ip_ ), INTENT( IN ) :: got, expected
   CHARACTER( * ), INTENT( IN ) :: label
   CHARACTER( LEN = 40 ) :: buf
   IF ( got == expected ) THEN
     CALL report( .TRUE., label )
   ELSE
     WRITE( buf, "( ' (flag ', I0, ', expected ', I0, ')' )" ) got, expected
     CALL report( .FALSE., label // TRIM( buf ) )
   END IF
   END SUBROUTINE report_flag

!  quiet controls: no library printing, so error tests do not clutter output

   SUBROUTINE quiet( control )
   TYPE( SLBLT_control_type ), INTENT( INOUT ) :: control
   control%print_level = - 1
   control%unit_error = - 1
   control%unit_warning = - 1
   control%unit_diagnostics = - 1
   END SUBROUTINE quiet

!  y = A x, A symmetric, lower triangle held by columns in (ptr,row,val);
!  entries with out-of-range rows are ignored, duplicates are summed

   SUBROUTINE symv( n, ptr, row, val, x, y )
   INTEGER( ip_ ), INTENT( IN ) :: n
   INTEGER( i8_ ), INTENT( IN ) :: ptr( : )
   INTEGER( ip_ ), INTENT( IN ) :: row( : )
   REAL( rp_ ), INTENT( IN ) :: val( : ), x( : )
   REAL( rp_ ), INTENT( OUT ) :: y( : )
   INTEGER( ip_ ) :: i, j
   INTEGER( i8_ ) :: k
   y( 1 : n ) = zero
   DO j = 1, n
     DO k = ptr( j ), ptr( j + 1 ) - 1
       i = row( k )
       IF ( i < j .OR. i > n ) CYCLE
       y( i ) = y( i ) + val( k ) * x( j )
       IF ( i /= j ) y( j ) = y( j ) + val( k ) * x( i )
     END DO
   END DO
   END SUBROUTINE symv

!  infinity norm of A (symmetric, lower triangle stored)

   REAL( rp_ ) FUNCTION norm_a( n, ptr, row, val )
   INTEGER( ip_ ), INTENT( IN ) :: n
   INTEGER( i8_ ), INTENT( IN ) :: ptr( : )
   INTEGER( ip_ ), INTENT( IN ) :: row( : )
   REAL( rp_ ), INTENT( IN ) :: val( : )
   REAL( rp_ ) :: rs( n )
   INTEGER( ip_ ) :: i, j
   INTEGER( i8_ ) :: k
   rs = zero
   DO j = 1, n
     DO k = ptr( j ), ptr( j + 1 ) - 1
       i = row( k )
       IF ( i < j .OR. i > n ) CYCLE
       rs( i ) = rs( i ) + ABS( val( k ) )
       IF ( i /= j ) rs( j ) = rs( j ) + ABS( val( k ) )
     END DO
   END DO
   norm_a = MAXVAL( rs )
   END FUNCTION norm_a

!  normwise backward error ||b - A x|| / ( ||A|| ||x|| + ||b|| )

   REAL( rp_ ) FUNCTION backward_error( n, ptr, row, val, x, b )
   INTEGER( ip_ ), INTENT( IN ) :: n
   INTEGER( i8_ ), INTENT( IN ) :: ptr( : )
   INTEGER( ip_ ), INTENT( IN ) :: row( : )
   REAL( rp_ ), INTENT( IN ) :: val( : ), x( : ), b( : )
   REAL( rp_ ) :: r( n ), den
   CALL symv( n, ptr, row, val, x, r )
   r = b( 1 : n ) - r
   den = norm_a( n, ptr, row, val ) * MAXVAL( ABS( x( 1 : n ) ) )              &
         + MAXVAL( ABS( b( 1 : n ) ) )
   IF ( den == zero ) THEN
     backward_error = MAXVAL( ABS( r ) )
   ELSE
     backward_error = MAXVAL( ABS( r ) ) / den
   END IF
   END FUNCTION backward_error

!  a reproducible "random" solution vector with entries in [-3.5,3.5]

   SUBROUTINE make_solution( n, x )
   INTEGER( ip_ ), INTENT( IN ) :: n
   REAL( rp_ ), INTENT( OUT ) :: x( : )
   INTEGER( ip_ ) :: i
   DO i = 1, n
     x( i ) = REAL( MOD( 7 * i, 8 ), rp_ ) - 3.5_rp_
   END DO
   END SUBROUTINE make_solution

!  tridiagonal matrix with diagonal d and off-diagonal o (lower, by columns).
!  alternate = .TRUE. makes the diagonal alternate in sign (indefinite)

   SUBROUTINE tridiag( n, d, o, alternate, ptr, row, val )
   INTEGER( ip_ ), INTENT( IN ) :: n
   REAL( rp_ ), INTENT( IN ) :: d, o
   LOGICAL, INTENT( IN ) :: alternate
   INTEGER( i8_ ), ALLOCATABLE, INTENT( OUT ) :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE, INTENT( OUT ) :: row( : )
   REAL( rp_ ), ALLOCATABLE, INTENT( OUT ) :: val( : )
   INTEGER( ip_ ) :: j
   INTEGER( i8_ ) :: k
   ALLOCATE( ptr( n + 1 ), row( MAX( 2 * n - 1, 1 ) ),                        &
             val( MAX( 2 * n - 1, 1 ) ) )
   k = 1
   DO j = 1, n
     ptr( j ) = k
     row( k ) = j
     val( k ) = d
     IF ( alternate .AND. MOD( j, 2 ) == 0 ) val( k ) = - d
     k = k + 1
     IF ( j < n ) THEN
       row( k ) = j + 1 ; val( k ) = o ; k = k + 1
     END IF
   END DO
   ptr( n + 1 ) = k
   END SUBROUTINE tridiag

!  saddle-point (KKT) matrix [ H  J^T ; J  0 ] with H = tridiag(-1,4,-1)
!  of order m and J (p x m, p < m) with J(i,i) = 1, J(i,i+1) = 1.
!  The (2,2) block has explicit zero diagonal entries, so 2x2 pivots and/or
!  delayed pivots are required

   SUBROUTINE kkt( m, p, ptr, row, val )
   INTEGER( ip_ ), INTENT( IN ) :: m, p
   INTEGER( i8_ ), ALLOCATABLE, INTENT( OUT ) :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE, INTENT( OUT ) :: row( : )
   REAL( rp_ ), ALLOCATABLE, INTENT( OUT ) :: val( : )
   INTEGER( ip_ ) :: j, n
   INTEGER( i8_ ) :: k
   n = m + p
   ALLOCATE( ptr( n + 1 ), row( 4 * n ), val( 4 * n ) )
   k = 1
   DO j = 1, m
     ptr( j ) = k
     row( k ) = j ; val( k ) = 4.0_rp_ ; k = k + 1
     IF ( j < m ) THEN
       row( k ) = j + 1 ; val( k ) = - one ; k = k + 1
     END IF
     IF ( j - 1 >= 1 .AND. j - 1 <= p ) THEN   ! J(j-1,j) = 1
       row( k ) = m + j - 1 ; val( k ) = one ; k = k + 1
     END IF
     IF ( j <= p ) THEN                        ! J(j,j) = 1
       row( k ) = m + j ; val( k ) = one ; k = k + 1
     END IF
   END DO
   DO j = m + 1, n
     ptr( j ) = k
     row( k ) = j ; val( k ) = zero ; k = k + 1
   END DO
   ptr( n + 1 ) = k
   END SUBROUTINE kkt

!  run analyse/factor/solve with given controls and check flags and accuracy.
!  If expected_flag_analyse/factor are given, the corresponding phase is
!  expected to return exactly that flag (and later phases are skipped if it
!  is negative); otherwise any non-negative flag is accepted. The solve is
!  checked for 1 and for 3 right-hand sides (the latter with ldx > n)

   SUBROUTINE full_solve( label, n, ptr, row, val, control, posdef, check,    &
                          order, scale, etol, expected_flag_factor,          &
                          inform_out, finite_only )
   CHARACTER( * ), INTENT( IN ) :: label
   INTEGER( ip_ ), INTENT( IN ) :: n
   INTEGER( i8_ ), INTENT( IN ) :: ptr( : )
   INTEGER( ip_ ), INTENT( IN ) :: row( : )
   REAL( rp_ ), INTENT( IN ) :: val( : )
   TYPE( SLBLT_control_type ), INTENT( IN ) :: control
   LOGICAL, INTENT( IN ) :: posdef, check
   INTEGER( ip_ ), OPTIONAL, INTENT( IN ) :: order( : )
   REAL( rp_ ), OPTIONAL, INTENT( INOUT ) :: scale( : )
   REAL( rp_ ), OPTIONAL, INTENT( IN ) :: etol
   INTEGER( ip_ ), OPTIONAL, INTENT( IN ) :: expected_flag_factor
   TYPE( SLBLT_inform_type ), OPTIONAL, INTENT( OUT ) :: inform_out
   LOGICAL, OPTIONAL, INTENT( IN ) :: finite_only

   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( ip_ ) :: flag, j, ldx
   INTEGER( ip_ ), ALLOCATABLE :: ord( : )
   REAL( rp_ ), ALLOCATABLE :: xs( : ), b( : ), x( : ), xm( :, : ), bm( :, : )
   REAL( rp_ ) :: be, bem, t
   LOGICAL :: ok
   CHARACTER( LEN = 60 ) :: buf

   t = tol ; IF ( PRESENT( etol ) ) t = etol
   ALLOCATE( xs( n ), b( n ), x( n ) )
   CALL make_solution( n, xs )
   CALL symv( n, ptr, row, val, xs, b )

!  analyse

   IF ( PRESENT( order ) ) THEN
     ALLOCATE( ord( n ) ) ; ord = order( 1 : n )
     CALL SLBLT_analyse( check, n, ptr, row, akeep, control, inform,           &
                         val = val, order = ord )
   ELSE
     CALL SLBLT_analyse( check, n, ptr, row, akeep, control, inform,           &
                         val = val )
   END IF
   IF ( PRESENT( inform_out ) ) inform_out = inform
   IF ( inform%flag == SLBLT_ERROR_NO_METIS ) THEN
     CALL report( .TRUE., label // ' (METIS unavailable, skipped)' )
     GO TO 900
   END IF
   IF ( inform%flag < 0 ) THEN
     WRITE( buf, "( ' (analyse flag ', I0, ')' )" ) inform%flag
     CALL report( .FALSE., label // TRIM( buf ) )
     GO TO 900
   END IF

!  factorize

   IF ( PRESENT( scale ) ) THEN
     CALL SLBLT_factor( posdef, val, akeep, fkeep, control, inform,            &
                        scale = scale, ptr = ptr, row = row )
   ELSE
     CALL SLBLT_factor( posdef, val, akeep, fkeep, control, inform,            &
                        ptr = ptr, row = row )
   END IF
   IF ( PRESENT( inform_out ) ) inform_out = inform
   IF ( PRESENT( expected_flag_factor ) ) THEN
     IF ( inform%flag /= expected_flag_factor ) THEN
       WRITE( buf, "( ' (factor flag ', I0, ', expected ', I0, ')' )" )        &
         inform%flag, expected_flag_factor
       CALL report( .FALSE., label // TRIM( buf ) )
       GO TO 900
     END IF
     IF ( expected_flag_factor < 0 ) THEN
       CALL report( .TRUE., label )
       GO TO 900
     END IF
   ELSE IF ( inform%flag < 0 ) THEN
     WRITE( buf, "( ' (factor flag ', I0, ')' )" ) inform%flag
     CALL report( .FALSE., label // TRIM( buf ) )
     GO TO 900
   END IF

!  solve, one rhs

   x = b
   CALL SLBLT_solve( x, akeep, fkeep, control, inform )
   IF ( inform%flag < 0 ) THEN
     WRITE( buf, "( ' (solve flag ', I0, ')' )" ) inform%flag
     CALL report( .FALSE., label // TRIM( buf ) )
     GO TO 900
   END IF
   be = backward_error( n, ptr, row, val, x, b )

!  solve, three rhs (b, 2b, -b) with ldx = n + 2

   ldx = n + 2
   ALLOCATE( xm( ldx, 3 ), bm( ldx, 3 ) )
   bm = zero
   bm( 1 : n, 1 ) = b ; bm( 1 : n, 2 ) = 2.0_rp_ * b ; bm( 1 : n, 3 ) = - b
   xm = bm
   CALL SLBLT_solve( 3_ip_, xm, ldx, akeep, fkeep, control, inform )
   bem = zero
   IF ( inform%flag >= 0 ) THEN
     DO j = 1, 3
       bem = MAX( bem, backward_error( n, ptr, row, val, xm( 1 : n, j ),       &
                                       bm( 1 : n, j ) ) )
     END DO
   ELSE
     bem = HUGE( one )
   END IF
   ok = be <= t .AND. bem <= t
   IF ( PRESENT( finite_only ) ) THEN
     IF ( finite_only ) ok = ALL( ABS( x ) <= HUGE( one ) ) .AND.              &
                             ALL( ABS( xm( 1 : n, : ) ) <= HUGE( one ) )
   END IF
   IF ( ok ) THEN
     CALL report( .TRUE., label )
   ELSE
     WRITE( buf, "( ' (backward error ', ES9.2, ', ', ES9.2, ')' )" ) be, bem
     CALL report( .FALSE., label // TRIM( buf ) )
   END IF

900 CONTINUE
   CALL SLBLT_free( akeep, fkeep, flag )
   END SUBROUTINE full_solve

!  =========================================================================
!  invalid arguments and call sequence
!  =========================================================================

   SUBROUTINE case_n_negative( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ) :: ptr( 1 ) = (/ 1 /)
   INTEGER( ip_ ) :: row( 1 ) = (/ 1 /), flag
   REAL( rp_ ) :: val( 1 ) = (/ one /), x( 1 )
   CALL quiet( control )
   CALL SLBLT_analyse( .TRUE., - 1_ip_, ptr, row, akeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_A_N_OOR, 'analyse n = -1' )
   CALL SLBLT_analyse( .FALSE., - HUGE( 1_ip_ ), ptr, row, akeep, control,     &
                       inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_A_N_OOR, 'analyse n = -HUGE' )
   CALL SLBLT_analyse_coord( - 5_ip_, 1_long_, row, row, akeep, control,       &
                             inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_A_N_OOR, 'analyse_coord n = -5' )

!  factor and solve must refuse to work after a failed analyse

   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_CALL_SEQUENCE,                   &
                     'factor after failed analyse' )
   x = one
   CALL SLBLT_solve( x, akeep, fkeep, control, inform )
   CALL report( inform%flag < 0, 'solve after failed analyse' )
   CALL SLBLT_free( akeep, fkeep, flag )
   END SUBROUTINE case_n_negative

   SUBROUTINE case_n_zero( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ) :: ptr( 1 ) = (/ 1 /)
   INTEGER( ip_ ) :: row( 1 ) = (/ 1 /), flag
   REAL( rp_ ) :: val( 1 ) = (/ one /), x( 1 ), x2( 1, 2 )
   CALL quiet( control )
   CALL SLBLT_analyse( .TRUE., 0_ip_, ptr, row, akeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_SUCCESS, 'analyse n = 0 (check)' )
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_SUCCESS, 'factor n = 0 indefinite' )
   CALL report( inform%matrix_rank == 0, 'factor n = 0 gives rank 0' )
   x = 42.0_rp_
   CALL SLBLT_solve( x, akeep, fkeep, control, inform )
   CALL report( inform%flag == SLBLT_SUCCESS .AND. x( 1 ) == 42.0_rp_,         &
                'solve n = 0 leaves x untouched' )
   x2 = 7.0_rp_
   CALL SLBLT_solve( 2_ip_, x2, 1_ip_, akeep, fkeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_SUCCESS, 'solve n = 0, nrhs = 2' )
   CALL SLBLT_factor( .TRUE., val, akeep, fkeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_SUCCESS, 'factor n = 0 posdef' )
   CALL SLBLT_free( akeep, fkeep, flag )
   CALL report_flag( flag, 0_ip_, 'free after n = 0' )

!  coordinate analyse with n = 0 and ne = 0

   CALL SLBLT_analyse_coord( 0_ip_, 0_long_, row, row, akeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_SUCCESS, 'analyse_coord n = 0, ne = 0' )
   CALL SLBLT_free( akeep, fkeep, flag )
   END SUBROUTINE case_n_zero

   SUBROUTINE case_bad_ordering_value( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   INTEGER( ip_ ) :: flag
   CALL tridiag( 5_ip_, 4.0_rp_, one, .FALSE., ptr, row, val )
   CALL quiet( control )
   control%ordering = - 1
   CALL SLBLT_analyse( .TRUE., 5_ip_, ptr, row, akeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_ORDER, 'ordering = -1' )
   control%ordering = 3
   CALL SLBLT_analyse( .TRUE., 5_ip_, ptr, row, akeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_ORDER, 'ordering = 3' )
   control%ordering = HUGE( 1_ip_ )
   CALL SLBLT_analyse( .TRUE., 5_ip_, ptr, row, akeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_ORDER, 'ordering = HUGE' )
   CALL SLBLT_free( akeep, fkeep, flag )
   END SUBROUTINE case_bad_ordering_value

   SUBROUTINE case_bad_user_order( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   INTEGER( ip_ ) :: order( 5 ), short( 3 ), flag
   CALL tridiag( 5_ip_, 4.0_rp_, one, .FALSE., ptr, row, val )
   CALL quiet( control )
   control%ordering = 0
   CALL SLBLT_analyse( .TRUE., 5_ip_, ptr, row, akeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_ORDER, 'ordering = 0, no order' )
   order = (/ 1, 2, 2, 4, 5 /)
   CALL SLBLT_analyse( .TRUE., 5_ip_, ptr, row, akeep, control, inform,        &
                       order = order )
   CALL report_flag( inform%flag, SLBLT_ERROR_ORDER, 'order with duplicate' )
   order = (/ 1, 2, 3, 4, 6 /)
   CALL SLBLT_analyse( .TRUE., 5_ip_, ptr, row, akeep, control, inform,        &
                       order = order )
   CALL report_flag( inform%flag, SLBLT_ERROR_ORDER, 'order entry > n' )
   order = (/ 0, 2, 3, 4, 5 /)
   CALL SLBLT_analyse( .TRUE., 5_ip_, ptr, row, akeep, control, inform,        &
                       order = order )
   CALL report_flag( inform%flag, SLBLT_ERROR_ORDER, 'order entry = 0' )
   short = (/ 1, 2, 3 /)
   CALL SLBLT_analyse( .TRUE., 5_ip_, ptr, row, akeep, control, inform,        &
                       order = short )
   CALL report_flag( inform%flag, SLBLT_ERROR_ORDER, 'order shorter than n' )

!  negative entries are allowed (their absolute values are used)

   order = (/ - 5, 4, - 3, 2, - 1 /)
   CALL SLBLT_free( akeep, fkeep, flag )
   CALL full_solve( 'order with negative entries accepted', 5_ip_, ptr, row,  &
                    val, control, .FALSE., .TRUE., order = order )
   END SUBROUTINE case_bad_user_order

   SUBROUTINE case_ordering2_no_val( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   INTEGER( ip_ ) :: flag
   CALL tridiag( 5_ip_, 4.0_rp_, one, .FALSE., ptr, row, val )
   CALL quiet( control )
   control%ordering = 2
   CALL SLBLT_analyse( .TRUE., 5_ip_, ptr, row, akeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_VAL, 'ordering = 2 without val' )
   CALL SLBLT_free( akeep, fkeep, flag )
   END SUBROUTINE case_ordering2_no_val

   SUBROUTINE case_bad_ptr( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ) :: ptr( 4 )
   INTEGER( ip_ ) :: row( 3 ) = (/ 1, 2, 3 /), flag
   CALL quiet( control )
   ptr = (/ 1_i8_, 3_i8_, 2_i8_, 4_i8_ /)      ! decreasing
   CALL SLBLT_analyse( .TRUE., 3_ip_, ptr, row, akeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_A_PTR, 'decreasing ptr' )
   ptr = (/ 0_i8_, 1_i8_, 2_i8_, 3_i8_ /)      ! ptr(1) /= 1
   CALL SLBLT_analyse( .TRUE., 3_ip_, ptr, row, akeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_A_PTR, 'ptr(1) = 0' )
   CALL SLBLT_free( akeep, fkeep, flag )
   END SUBROUTINE case_bad_ptr

   SUBROUTINE case_oor_and_dup( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ) :: ptr( 4 )
   INTEGER( ip_ ) :: row( 7 ), flag
   REAL( rp_ ) :: val( 7 ), x( 3 ), b( 3 ), xs( 3 )
   INTEGER( i8_ ) :: ptrc( 4 )
   INTEGER( ip_ ) :: rowc( 5 )
   REAL( rp_ ) :: valc( 5 )
   CALL quiet( control )

!  A = [4 1 0; 1 4 1; 0 1 4]; clean form for residual checks

   ptrc = (/ 1_i8_, 3_i8_, 5_i8_, 6_i8_ /)
   rowc = (/ 1, 2, 2, 3, 3 /)
   valc = (/ 4.0_rp_, one, 4.0_rp_, one, 4.0_rp_ /)
   xs = (/ one, - 2.0_rp_, 3.0_rp_ /)
   CALL symv( 3_ip_, ptrc, rowc, valc, xs, b )

!  out-of-range rows (0 and 99) are ignored with a warning

   ptr = (/ 1_i8_, 4_i8_, 6_i8_, 8_i8_ /)
   row = (/ 1, 2, 99, 2, 3, 3, 0 /)
   val = (/ 4.0_rp_, one, 1.0E+3_rp_, 4.0_rp_, one, 4.0_rp_, 5.0_rp_ /)
   CALL SLBLT_analyse( .TRUE., 3_ip_, ptr, row, akeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_WARNING_IDX_OOR,                       &
                     'out-of-range rows: warning' )
   CALL report( inform%matrix_outrange == 2, 'out-of-range rows counted' )
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   x = b
   CALL SLBLT_solve( x, akeep, fkeep, control, inform )
   CALL report( inform%flag >= 0 .AND.                                         &
                backward_error( 3_ip_, ptrc, rowc, valc, x, b ) <= tol,        &
                'out-of-range rows: entries ignored in solve' )

!  duplicates (entry (2,1) split into 0.25 + 0.75) are summed with a warning

   ptr = (/ 1_i8_, 4_i8_, 6_i8_, 7_i8_ /)
   row = (/ 1, 2, 2, 2, 3, 3, 0 /)
   val = (/ 4.0_rp_, 0.25_rp_, 0.75_rp_, 4.0_rp_, one, 4.0_rp_, zero /)
   CALL SLBLT_analyse( .TRUE., 3_ip_, ptr, row, akeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_WARNING_DUP_IDX, 'duplicates: warning' )
   CALL report( inform%matrix_dup == 1, 'duplicates counted' )
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   x = b
   CALL SLBLT_solve( x, akeep, fkeep, control, inform )
   CALL report( inform%flag >= 0 .AND.                                         &
                backward_error( 3_ip_, ptrc, rowc, valc, x, b ) <= tol,        &
                'duplicates: summed in solve' )

!  both duplicates and out-of-range entries

   ptr = (/ 1_i8_, 4_i8_, 7_i8_, 8_i8_ /)
   row = (/ 1, 2, 2, 2, 3, - 4, 3 /)
   val = (/ 4.0_rp_, 0.5_rp_, 0.5_rp_, 4.0_rp_, one, 9.0_rp_, 4.0_rp_ /)
   CALL SLBLT_analyse( .TRUE., 3_ip_, ptr, row, akeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_WARNING_DUP_AND_OOR,                   &
                     'duplicates and out-of-range: warning' )
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   x = b
   CALL SLBLT_solve( x, akeep, fkeep, control, inform )
   CALL report( inform%flag >= 0 .AND.                                         &
                backward_error( 3_ip_, ptrc, rowc, valc, x, b ) <= tol,        &
                'duplicates and out-of-range: solve' )
   CALL SLBLT_free( akeep, fkeep, flag )
   END SUBROUTINE case_oor_and_dup

   SUBROUTINE case_all_oor( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ) :: ptr( 3 ) = (/ 1_i8_, 2_i8_, 3_i8_ /)
   INTEGER( ip_ ) :: row( 2 ) = (/ 7, - 1 /), flag
   CALL quiet( control )
   CALL SLBLT_analyse( .TRUE., 2_ip_, ptr, row, akeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_A_ALL_OOR,                       &
                     'all entries out of range' )
   CALL SLBLT_free( akeep, fkeep, flag )
   END SUBROUTINE case_all_oor

   SUBROUTINE case_call_sequence( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   REAL( rp_ ) :: x( 5 ), d( 2, 5 )
   INTEGER( ip_ ) :: piv( 5 ), flag
   CALL tridiag( 5_ip_, 4.0_rp_, one, .FALSE., ptr, row, val )
   CALL quiet( control )
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_CALL_SEQUENCE,                   &
                     'factor before analyse' )
   CALL SLBLT_analyse( .TRUE., 5_ip_, ptr, row, akeep, control, inform )
   x = one
   CALL SLBLT_solve( x, akeep, fkeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_CALL_SEQUENCE,                   &
                     'solve before factor' )
   CALL SLBLT_enquire_indef( akeep, fkeep, control, inform, piv_order = piv,   &
                             d = d )
   CALL report_flag( inform%flag, SLBLT_ERROR_CALL_SEQUENCE,                   &
                     'enquire_indef before factor' )
   CALL SLBLT_enquire_posdef( akeep, fkeep, control, inform, d( 1, : ) )
   CALL report_flag( inform%flag, SLBLT_ERROR_CALL_SEQUENCE,                   &
                     'enquire_posdef before factor' )
   d = one
   CALL SLBLT_alter( d, akeep, fkeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_CALL_SEQUENCE,                   &
                     'alter before factor' )

!  free twice must be harmless

   CALL SLBLT_free( akeep, fkeep, flag )
   CALL SLBLT_free( akeep, fkeep, flag )
   CALL report_flag( flag, 0_ip_, 'double free' )
   END SUBROUTINE case_call_sequence

   SUBROUTINE case_nocheck_needs_ptr_row( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   INTEGER( ip_ ) :: flag
   CALL tridiag( 5_ip_, 4.0_rp_, one, .FALSE., ptr, row, val )
   CALL quiet( control )
   CALL SLBLT_analyse( .FALSE., 5_ip_, ptr, row, akeep, control, inform )
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_PTR_ROW,                         &
                     'check = .false. and factor without ptr/row' )
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform, ptr = ptr )
   CALL report_flag( inform%flag, SLBLT_ERROR_PTR_ROW,                         &
                     'check = .false. and factor without row' )
   CALL SLBLT_free( akeep, fkeep, flag )
   END SUBROUTINE case_nocheck_needs_ptr_row

   SUBROUTINE case_solve_args( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   REAL( rp_ ) :: x( 5 ), x4( 4 ), xm( 5, 2 ), b( 5 ), xs( 5 )
   INTEGER( ip_ ) :: flag, job
   CHARACTER( LEN = 40 ) :: lab
   CALL tridiag( 5_ip_, 4.0_rp_, one, .TRUE., ptr, row, val )
   CALL quiet( control )
   CALL SLBLT_analyse( .TRUE., 5_ip_, ptr, row, akeep, control, inform )
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_SUCCESS, 'indefinite tridiag factor' )
   x = one
   CALL SLBLT_solve( x, akeep, fkeep, control, inform, job = 0_ip_ )
   CALL report_flag( inform%flag, SLBLT_ERROR_JOB_OOR, 'solve job = 0' )
   CALL SLBLT_solve( x, akeep, fkeep, control, inform, job = 5_ip_ )
   CALL report_flag( inform%flag, SLBLT_ERROR_JOB_OOR, 'solve job = 5' )
   CALL SLBLT_solve( x, akeep, fkeep, control, inform, job = - HUGE( 1_ip_ ) )
   CALL report_flag( inform%flag, SLBLT_ERROR_JOB_OOR, 'solve job = -HUGE' )
   x4 = one
   CALL SLBLT_solve( x4, akeep, fkeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_X_SIZE, 'solve with size(x) < n' )
   CALL SLBLT_solve( 0_ip_, xm, 5_ip_, akeep, fkeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_X_SIZE, 'solve nrhs = 0' )
   CALL SLBLT_solve( - 1_ip_, xm, 5_ip_, akeep, fkeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_X_SIZE, 'solve nrhs = -1' )
   CALL SLBLT_solve( 1_ip_, xm, 4_ip_, akeep, fkeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_X_SIZE, 'solve ldx = n - 1' )

!  partial solves: 1 then 2 then 3, and 1 then 4, must equal a full solve

   CALL make_solution( 5_ip_, xs )
   CALL symv( 5_ip_, ptr, row, val, xs, b )
   x = b
   DO job = 1, 3
     CALL SLBLT_solve( x, akeep, fkeep, control, inform, job = job )
   END DO
   CALL report( inform%flag == 0 .AND.                                         &
                backward_error( 5_ip_, ptr, row, val, x, b ) <= tol,           &
                'partial solves job = 1, 2, 3' )
   x = b
   CALL SLBLT_solve( x, akeep, fkeep, control, inform, job = 1_ip_ )
   CALL SLBLT_solve( x, akeep, fkeep, control, inform, job = 4_ip_ )
   CALL report( inform%flag == 0 .AND.                                         &
                backward_error( 5_ip_, ptr, row, val, x, b ) <= tol,           &
                'partial solves job = 1, 4' )

!  posdef factor: diagonal jobs 2 and 4 are not allowed

   CALL tridiag( 5_ip_, 4.0_rp_, - one, .FALSE., ptr, row, val )
   CALL SLBLT_analyse( .TRUE., 5_ip_, ptr, row, akeep, control, inform )
   CALL SLBLT_factor( .TRUE., val, akeep, fkeep, control, inform )
   DO job = 2, 4, 2
     x = one
     CALL SLBLT_solve( x, akeep, fkeep, control, inform, job = job )
     WRITE( lab, "( 'posdef solve job = ', I0, ' rejected' )" ) job
     CALL report_flag( inform%flag, SLBLT_ERROR_JOB_OOR, TRIM( lab ) )
   END DO
   CALL make_solution( 5_ip_, xs )
   CALL symv( 5_ip_, ptr, row, val, xs, b )
   x = b
   CALL SLBLT_solve( x, akeep, fkeep, control, inform, job = 1_ip_ )
   CALL SLBLT_solve( x, akeep, fkeep, control, inform, job = 3_ip_ )
   CALL report( inform%flag == 0 .AND.                                         &
                backward_error( 5_ip_, ptr, row, val, x, b ) <= tol,           &
                'posdef partial solves job = 1, 3' )
   CALL SLBLT_free( akeep, fkeep, flag )
   END SUBROUTINE case_solve_args

!  partial solves with a positive-definite factor A = L L^T: job = 1 must
!  apply L^{-1} on its own, which is checked through the identity
!  || L^{-1} b ||^2 = b^T A^{-1} b (also valid with scaling, since then
!  S A S = L L^T), and job = 1 then job = 3 must solve A x = b. Checked for
!  3 right-hand sides, with and without scaling, and with the tree in one
!  or several subtrees

   SUBROUTINE case_posdef_partial_solves( )
   INTEGER( ip_ ), PARAMETER :: n = 500, nrhs = 3
   INTEGER( ip_ ), PARAMETER :: scalings( 3 ) = (/ 0, 1, 4 /)
   REAL, PARAMETER :: imbalance( 2 ) = (/ 1.2, 1.0 /)
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   REAL( rp_ ) :: xs( n ), b( n, nrhs ), x( n, nrhs ), y( n, nrhs )
   REAL( rp_ ) :: yy, bx, be
   INTEGER( ip_ ) :: flag, is, ii, r
   LOGICAL :: ok
   CHARACTER( LEN = 60 ) :: lab
   CALL quiet( control )
   CALL tridiag( n, 4.0_rp_, - one, .FALSE., ptr, row, val )
   CALL make_solution( n, xs )
   DO r = 1, nrhs
     CALL symv( n, ptr, row, val, xs * REAL( r, rp_ ), b( :, r ) )
   END DO
   control%nemin = 128
   DO ii = 1, 2
     control%max_load_inbalance = imbalance( ii )
     DO is = 1, 3
       control%scaling = scalings( is )
       WRITE( lab, "( 'posdef partial solves, scaling = ', I0, ', ', A )" )    &
         scalings( is ), TRIM( MERGE( 'one subtree', 'split tree ', ii == 1 ) )
       CALL SLBLT_analyse( .TRUE., n, ptr, row, akeep, control, inform )
       IF ( ii == 2 .AND. is == 1 )                                           &
         CALL report( inform%nparts > 1, TRIM( lab ) // ': tree is split' )
       CALL SLBLT_factor( .TRUE., val, akeep, fkeep, control, inform )
       ok = inform%flag == 0

!  full solve, and job = 1 alone

       x = b
       CALL SLBLT_solve( nrhs, x, n, akeep, fkeep, control, inform )
       ok = ok .AND. inform%flag == 0
       y = b
       CALL SLBLT_solve( nrhs, y, n, akeep, fkeep, control, inform,            &
                         job = 1_ip_ )
       ok = ok .AND. inform%flag == 0
       DO r = 1, nrhs
         yy = DOT_PRODUCT( y( :, r ), y( :, r ) )
         bx = DOT_PRODUCT( b( :, r ), x( :, r ) )
         ok = ok .AND. ABS( yy - bx ) <= tol * bx
       END DO
       CALL report( ok, TRIM( lab ) // ': job = 1' )

!  job = 3 applied to the result of job = 1

       CALL SLBLT_solve( nrhs, y, n, akeep, fkeep, control, inform,            &
                         job = 3_ip_ )
       ok = inform%flag == 0
       DO r = 1, nrhs
         be = backward_error( n, ptr, row, val, y( :, r ), b( :, r ) )
         ok = ok .AND. be <= tol
       END DO
       CALL report( ok, TRIM( lab ) // ': job = 1, 3' )
       CALL SLBLT_free( akeep, fkeep, flag )
     END DO
   END DO
   END SUBROUTINE case_posdef_partial_solves

   SUBROUTINE case_enquire_mismatch( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   REAL( rp_ ) :: d( 2, 5 ), dp( 5 ), x( 5 ), b( 5 ), xs( 5 )
   INTEGER( ip_ ) :: piv( 5 ), flag
   CALL tridiag( 5_ip_, 4.0_rp_, - one, .FALSE., ptr, row, val )
   CALL quiet( control )

!  posdef factorization: enquire_indef and alter must be rejected

   CALL SLBLT_analyse( .TRUE., 5_ip_, ptr, row, akeep, control, inform )
   CALL SLBLT_factor( .TRUE., val, akeep, fkeep, control, inform )
   CALL SLBLT_enquire_indef( akeep, fkeep, control, inform, piv_order = piv,   &
                             d = d )
   CALL report_flag( inform%flag, SLBLT_ERROR_NOT_LDLT,                        &
                     'enquire_indef after posdef factor' )
   d = one
   CALL SLBLT_alter( d, akeep, fkeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_NOT_LDLT,                        &
                     'alter after posdef factor' )
   CALL SLBLT_enquire_posdef( akeep, fkeep, control, inform, dp )
   CALL report( inform%flag == 0 .AND. ALL( dp > zero ),                       &
                'enquire_posdef gives positive diagonal' )

!  indefinite factorization: enquire_posdef must be rejected

   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   CALL SLBLT_enquire_posdef( akeep, fkeep, control, inform, dp )
   CALL report_flag( inform%flag, SLBLT_ERROR_NOT_LLT,                         &
                     'enquire_posdef after indefinite factor' )

!  enquire then alter with the same D must leave the solution unchanged

   CALL SLBLT_enquire_indef( akeep, fkeep, control, inform, piv_order = piv,   &
                             d = d )
   CALL report( inform%flag == 0, 'enquire_indef after indefinite factor' )
   CALL SLBLT_alter( d, akeep, fkeep, control, inform )
   CALL make_solution( 5_ip_, xs )
   CALL symv( 5_ip_, ptr, row, val, xs, b )
   x = b
   CALL SLBLT_solve( x, akeep, fkeep, control, inform )
   CALL report( inform%flag == 0 .AND.                                         &
                backward_error( 5_ip_, ptr, row, val, x, b ) <= tol,           &
                'alter with unchanged D keeps solution' )
   CALL SLBLT_free( akeep, fkeep, flag )
   END SUBROUTINE case_enquire_mismatch

   SUBROUTINE case_no_saved_scaling( )
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   CALL tridiag( 6_ip_, 4.0_rp_, one, .TRUE., ptr, row, val )
   CALL quiet( control )
   control%scaling = 3                       ! needs ordering = 2
   control%ordering = 1
   CALL full_solve( 'scaling = 3 without matching ordering', 6_ip_, ptr, row, &
                    val, control, .FALSE., .TRUE.,                             &
                    expected_flag_factor = SLBLT_ERROR_NO_SAVED_SCALING )

!  ordering = 2 with scaling /= 3 raises a warning but still solves

   control%ordering = 2
   control%scaling = 0
   CALL full_solve( 'ordering = 2 with scaling = 0 (warning 8)', 6_ip_, ptr,  &
                    row, val, control, .FALSE., .TRUE.,                        &
                    expected_flag_factor = SLBLT_WARNING_MATCH_ORD_NO_SCALE )
   END SUBROUTINE case_no_saved_scaling

!  =========================================================================
!  extreme matrices
!  =========================================================================

   SUBROUTINE case_n_one( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ) :: ptr( 2 ) = (/ 1_i8_, 2_i8_ /)
   INTEGER( ip_ ) :: row( 1 ) = (/ 1 /), flag
   REAL( rp_ ) :: val( 1 ), x( 1 ), d( 2, 1 )
   INTEGER( ip_ ) :: ord1( 1 )
   CALL quiet( control )
   control%ordering = 0
   val = - 3.0_rp_
   CALL full_solve( 'n = 1, negative entry, indefinite', 1_ip_, ptr, row,     &
                    val, control, .FALSE., .TRUE., order = (/ 1_ip_ /) )
   val = 2.0_rp_
   CALL full_solve( 'n = 1, positive entry, posdef', 1_ip_, ptr, row,         &
                    val, control, .TRUE., .TRUE., order = (/ 1_ip_ /) )
   val = - 3.0_rp_
   CALL full_solve( 'n = 1, negative entry, posdef rejected', 1_ip_, ptr,     &
                    row, val, control, .TRUE., .TRUE., order = (/ 1_ip_ /),    &
                    expected_flag_factor = SLBLT_ERROR_NOT_POS_DEF )

!  inertia and D for n = 1

   val = - 3.0_rp_
   ord1 = 1
   CALL SLBLT_analyse( .TRUE., 1_ip_, ptr, row, akeep, control, inform,        &
                       order = ord1 )
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   CALL report( inform%num_neg == 1 .AND. inform%matrix_rank == 1,             &
                'n = 1 inertia: one negative pivot' )
   CALL SLBLT_enquire_indef( akeep, fkeep, control, inform, d = d )
   CALL report( ABS( d( 1, 1 ) + one / 3.0_rp_ ) <= tol .AND.                  &
                d( 2, 1 ) == zero, 'n = 1 enquire D = 1/a' )
   x = 6.0_rp_
   CALL SLBLT_solve( x, akeep, fkeep, control, inform )
   CALL report( ABS( x( 1 ) + 2.0_rp_ ) <= tol, 'n = 1 solve x = b/a' )
   CALL SLBLT_free( akeep, fkeep, flag )
   END SUBROUTINE case_n_one

   SUBROUTINE case_singular( )
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   INTEGER( ip_ ) :: flag
   REAL( rp_ ) :: x( 6 ), b( 6 ), xs( 6 )

!  numerically singular: tridiag(-1,2,-1) with A(1,1) = A(n,n) = 1 is the
!  graph Laplacian of a path, which has the constant null vector

   CALL tridiag( 6_ip_, 2.0_rp_, - one, .FALSE., ptr, row, val )
   val( 1 ) = one ; val( SIZE( val ) ) = one
   CALL quiet( control )
   control%action = .TRUE.
   CALL SLBLT_analyse( .TRUE., 6_ip_, ptr, row, akeep, control, inform )
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_WARNING_FACT_SINGULAR,                 &
                     'singular Laplacian, action = .true.' )
   CALL report( inform%matrix_rank == 5, 'singular Laplacian rank = n - 1' )
   CALL make_solution( 6_ip_, xs )
   CALL symv( 6_ip_, ptr, row, val, xs, b )
   x = b
   CALL SLBLT_solve( x, akeep, fkeep, control, inform )
   CALL report( inform%flag >= 0 .AND. ALL( ABS( x ) <= HUGE( one ) ) .AND.    &
                backward_error( 6_ip_, ptr, row, val, x, b ) <= tol,           &
                'singular Laplacian, consistent rhs solved' )
   control%action = .FALSE.
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_SINGULAR,                        &
                     'singular Laplacian, action = .false.' )
   CALL SLBLT_free( akeep, fkeep, flag )

!  all-zero matrix

   val = zero
   control%action = .TRUE.
   CALL SLBLT_analyse( .TRUE., 6_ip_, ptr, row, akeep, control, inform )
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   CALL report( inform%flag == SLBLT_WARNING_FACT_SINGULAR .AND.               &
                inform%matrix_rank == 0, 'zero matrix: rank 0 warning' )
   CALL SLBLT_factor( .TRUE., val, akeep, fkeep, control, inform )
   CALL report_flag( inform%flag, SLBLT_ERROR_NOT_POS_DEF,                     &
                     'zero matrix: posdef rejected' )
   CALL SLBLT_free( akeep, fkeep, flag )
   END SUBROUTINE case_singular

   SUBROUTINE case_zero_diagonal_2x2( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   INTEGER( ip_ ) :: flag, piv( 2 )
   REAL( rp_ ) :: d( 2, 2 )
   INTEGER( i8_ ) :: p2( 3 ) = (/ 1_i8_, 3_i8_, 4_i8_ /)
   INTEGER( ip_ ) :: r2( 3 ) = (/ 1, 2, 2 /)
   REAL( rp_ ) :: v2( 3 ) = (/ zero, one, zero /)
   CALL quiet( control )

!  [0 1; 1 0] needs a single 2x2 pivot

   control%ordering = 0
   piv = (/ 1_ip_, 2_ip_ /)
   CALL SLBLT_analyse( .TRUE., 2_ip_, p2, r2, akeep, control, inform,          &
                       order = piv )
   CALL SLBLT_factor( .FALSE., v2, akeep, fkeep, control, inform )
   CALL report( inform%flag == 0 .AND. inform%num_two == 1 .AND.               &
                inform%num_neg == 1, '[0 1; 1 0]: one 2x2 pivot, inertia' )
   CALL SLBLT_enquire_indef( akeep, fkeep, control, inform, piv_order = piv,   &
                             d = d )
   CALL report( inform%flag == 0 .AND. piv( 1 ) < 0 .AND. piv( 2 ) < 0,        &
                '[0 1; 1 0]: piv_order flags a 2x2 block' )
   CALL SLBLT_free( akeep, fkeep, flag )
   CALL full_solve( '[0 1; 1 0] solve', 2_ip_, p2, r2, v2, control,           &
                    .FALSE., .TRUE., order = (/ 1_ip_, 2_ip_ /) )

!  KKT system with 40 zero diagonals

   control%ordering = 1
   CALL kkt( 60_ip_, 40_ip_, ptr, row, val )
   CALL full_solve( 'KKT 100x100 with zero (2,2) block', 100_ip_, ptr, row,   &
                    val, control, .FALSE., .TRUE., inform_out = inform )
   CALL report( inform%num_neg == 40, 'KKT inertia: 40 negative pivots' )
   END SUBROUTINE case_zero_diagonal_2x2

   SUBROUTINE case_extreme_scale( )
   TYPE( SLBLT_control_type ) :: control
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   REAL( rp_ ) :: big, tiny_s
   INTEGER( ip_ ) :: n, j
   INTEGER( i8_ ) :: k
   n = 50
   big = HUGE( one ) ** 0.25_rp_
   tiny_s = TINY( one ) ** 0.25_rp_
   CALL quiet( control )

!  uniformly huge entries

   CALL kkt( 30_ip_, 20_ip_, ptr, row, val )
   val = val * big
   CALL full_solve( 'KKT scaled by HUGE**0.25', n, ptr, row, val, control,    &
                    .FALSE., .TRUE. )

!  uniformly tiny entries: smaller than control%small, so with the default
!  small every pivot is treated as zero unless scaling is used, or small = 0

   CALL kkt( 30_ip_, 20_ip_, ptr, row, val )
   val = val * tiny_s
   IF ( tiny_s < control%small ) THEN
     control%action = .FALSE.
     CALL full_solve( 'KKT scaled by TINY**0.25, default small: singular',    &
                      n, ptr, row, val, control, .FALSE., .TRUE.,              &
                      expected_flag_factor = SLBLT_ERROR_SINGULAR )
     control%action = .TRUE.
   END IF
   control%small = zero
   CALL full_solve( 'KKT scaled by TINY**0.25, small = 0', n, ptr, row, val,  &
                    control, .FALSE., .TRUE. )
   control%small = 1.0E-20_rp_
   control%scaling = 4
   CALL full_solve( 'KKT scaled by TINY**0.25, equilibration scaling', n,     &
                    ptr, row, val, control, .FALSE., .TRUE. )
   control%scaling = 0

!  badly row/column scaled: D A D with D = diag( 10**(+-k) ) spread over
!  roughly sqrt(HUGE) .. 1/sqrt(HUGE); fixed only by scaling

   CALL tridiag( n, 4.0_rp_, one, .TRUE., ptr, row, val )
   DO j = 1, n
     DO k = ptr( j ), ptr( j + 1 ) - 1
       val( k ) = val( k ) * dscale( j, n ) * dscale( row( k ), n )
     END DO
   END DO
   DO j = 1, 4, 3                            ! Hungarian and equilibration
     control%scaling = j
     CALL full_solve( 'badly scaled tridiag, scaling = ' // CHAR( 48 + j ),   &
                      n, ptr, row, val, control, .FALSE., .TRUE. )
   END DO
   control%scaling = 0
   END SUBROUTINE case_extreme_scale

!  row/column scale factor for variable j: geometric spread between
!  HUGE**(1/8) and HUGE**(-1/8)

   REAL( rp_ ) FUNCTION dscale( j, n )
   INTEGER( ip_ ), INTENT( IN ) :: j, n
   REAL( rp_ ) :: e
   e = LOG( HUGE( one ) ) / 8.0_rp_
   dscale = EXP( e * ( 2.0_rp_ * REAL( j - 1, rp_ ) / REAL( n - 1, rp_ )       &
                 - one ) )
   END FUNCTION dscale

   SUBROUTINE case_not_posdef( )
   TYPE( SLBLT_control_type ) :: control
   INTEGER( ip_ ) :: i
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   CALL quiet( control )
   CALL tridiag( 20_ip_, 4.0_rp_, one, .TRUE., ptr, row, val )
   CALL full_solve( 'indefinite matrix with posdef = .true. rejected',        &
                    20_ip_, ptr, row, val, control, .TRUE., .TRUE.,            &
                    expected_flag_factor = SLBLT_ERROR_NOT_POS_DEF )

!  positive semi-definite (singular) matrix with posdef = .true.

!  (identity order, so the last pivot is exactly zero; with other orders
!  rounding may legitimately leave a tiny positive pivot)

   CALL tridiag( 20_ip_, 2.0_rp_, - one, .FALSE., ptr, row, val )
   val( 1 ) = one ; val( SIZE( val ) ) = one
   control%ordering = 0
   CALL full_solve( 'semidefinite matrix with posdef = .true. rejected',      &
                    20_ip_, ptr, row, val, control, .TRUE., .TRUE.,            &
                    order = (/ ( i, i = 1, 20 ) /),                            &
                    expected_flag_factor = SLBLT_ERROR_NOT_POS_DEF )
   END SUBROUTINE case_not_posdef

   SUBROUTINE case_large_n( )
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   INTEGER( ip_ ), PARAMETER :: n = 100000
   CALL quiet( control )
   CALL tridiag( n, 4.0_rp_, - one, .FALSE., ptr, row, val )
   CALL full_solve( 'n = 100000 SPD tridiag, posdef', n, ptr, row, val,       &
                    control, .TRUE., .TRUE., inform_out = inform )
   CALL report( inform%num_neg == 0 .AND. inform%matrix_rank == n,             &
                'n = 100000 SPD: rank n, no negative pivots' )
   CALL tridiag( n, 4.0_rp_, one, .TRUE., ptr, row, val )
   CALL full_solve( 'n = 100000 indefinite tridiag, check = .false.', n,      &
                    ptr, row, val, control, .FALSE., .FALSE.,                  &
                    inform_out = inform )
   CALL report( inform%num_neg == n / 2, 'n = 100000 indefinite inertia' )
   END SUBROUTINE case_large_n

   SUBROUTINE case_coord_input( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   INTEGER( ip_ ) :: irn( 8 ), jcn( 8 ), flag
   REAL( rp_ ) :: a( 8 ), x( 3 ), b( 3 ), xs( 3 )
   CALL quiet( control )

!  A = [4 1 0; 1 4 1; 0 1 4] given in coordinate form with upper-triangle
!  entries (which are mirrored), an out-of-range entry and a duplicate

   irn = (/ 1, 1, 2, 2, 3, 3, 9, 3 /)
   jcn = (/ 1, 2, 2, 3, 3, 3, 1, 2 /)
   a = (/ 4.0_rp_, one, 4.0_rp_, 0.5_rp_, 3.0_rp_, one, 7.0_rp_, 0.5_rp_ /)
   CALL SLBLT_analyse_coord( 3_ip_, 8_long_, irn, jcn, akeep, control, inform, &
                             val = a )
   CALL report( inform%flag == SLBLT_WARNING_DUP_AND_OOR .OR.                  &
                inform%flag == SLBLT_WARNING_DUP_IDX .OR.                      &
                inform%flag == SLBLT_WARNING_IDX_OOR,                          &
                'analyse_coord flags dup/out-of-range' )
   CALL report( inform%matrix_outrange == 1, 'analyse_coord counts 1 oor' )
   CALL SLBLT_factor( .FALSE., a, akeep, fkeep, control, inform )
   CALL tridiag( 3_ip_, 4.0_rp_, one, .FALSE., ptr, row, val )
   xs = (/ one, 2.0_rp_, - one /)
   CALL symv( 3_ip_, ptr, row, val, xs, b )
   x = b
   CALL SLBLT_solve( x, akeep, fkeep, control, inform )
   CALL report( inform%flag >= 0 .AND.                                         &
                backward_error( 3_ip_, ptr, row, val, x, b ) <= tol,           &
                'analyse_coord: upper entries mirrored, dups summed' )
   CALL SLBLT_free( akeep, fkeep, flag )
   END SUBROUTINE case_coord_input

!  32-bit ptr interface

   SUBROUTINE case_ptr32( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( i4_ ), ALLOCATABLE :: ptr32( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : ), x( : ), b( : ), xs( : )
   INTEGER( ip_ ) :: flag
   CALL quiet( control )
   CALL kkt( 12_ip_, 6_ip_, ptr, row, val )
   ALLOCATE( ptr32( SIZE( ptr ) ), x( 18 ), b( 18 ), xs( 18 ) )
   ptr32 = INT( ptr, i4_ )
   CALL SLBLT_analyse( .FALSE., 18_ip_, ptr32, row, akeep, control, inform )
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform,             &
                      ptr = ptr32, row = row )
   CALL make_solution( 18_ip_, xs )
   CALL symv( 18_ip_, ptr, row, val, xs, b )
   x = b
   CALL SLBLT_solve( x, akeep, fkeep, control, inform )
   CALL report( inform%flag >= 0 .AND.                                         &
                backward_error( 18_ip_, ptr, row, val, x, b ) <= tol,          &
                '32-bit ptr analyse/factor, check = .false.' )
   CALL SLBLT_free( akeep, fkeep, flag )
   END SUBROUTINE case_ptr32

!  =========================================================================
!  non-default control parameters
!  =========================================================================

   SUBROUTINE case_orderings( )
   TYPE( SLBLT_control_type ) :: control
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   INTEGER( ip_ ) :: order( 80 ), i
   CALL quiet( control )
   CALL kkt( 50_ip_, 30_ip_, ptr, row, val )
   control%ordering = 0
   order = (/ ( i, i = 1, 80 ) /)
   CALL full_solve( 'ordering = 0, identity order', 80_ip_, ptr, row, val,    &
                    control, .FALSE., .TRUE., order = order )
   order = (/ ( 81 - i, i = 1, 80 ) /)
   CALL full_solve( 'ordering = 0, reversed order', 80_ip_, ptr, row, val,    &
                    control, .FALSE., .TRUE., order = order )
   control%ordering = 1
   CALL full_solve( 'ordering = 1 (nested dissection)', 80_ip_, ptr, row,     &
                    val, control, .FALSE., .TRUE. )
   control%ordering = 2
   control%scaling = 3
   CALL full_solve( 'ordering = 2 (matching) + scaling = 3', 80_ip_, ptr,     &
                    row, val, control, .FALSE., .TRUE. )
   CALL full_solve( 'ordering = 2, check = .false.', 80_ip_, ptr, row, val,   &
                    control, .FALSE., .FALSE. )
   END SUBROUTINE case_orderings

   SUBROUTINE case_nemin( )
   TYPE( SLBLT_control_type ) :: control
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   INTEGER( ip_ ) :: k
   INTEGER( ip_ ), PARAMETER :: nem( 5 ) =                                     &
     (/ - HUGE( 1_ip_ ), 0_ip_, 1_ip_, 2_ip_, HUGE( 1_ip_ ) /)
   CHARACTER( LEN = 60 ) :: lab
   CALL quiet( control )
   CALL kkt( 200_ip_, 120_ip_, ptr, row, val )
   DO k = 1, SIZE( nem )
     control%nemin = nem( k )
     WRITE( lab, "( 'nemin = ', I0 )" ) nem( k )
     CALL full_solve( TRIM( lab ), 320_ip_, ptr, row, val, control,           &
                      .FALSE., .TRUE. )
   END DO
   END SUBROUTINE case_nemin

   SUBROUTINE case_scalings( )
   TYPE( SLBLT_control_type ) :: control
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   REAL( rp_ ) :: scale( 80 )
   INTEGER( ip_ ) :: k
   INTEGER( ip_ ), PARAMETER :: sc( 6 ) = (/ - 7, 1, 2, 4, 5, 1000 /)
   CHARACTER( LEN = 60 ) :: lab
   CALL quiet( control )
   CALL kkt( 50_ip_, 30_ip_, ptr, row, val )
   DO k = 1, SIZE( sc )
     control%scaling = sc( k )
     WRITE( lab, "( 'scaling = ', I0 )" ) sc( k )
     CALL full_solve( TRIM( lab ), 80_ip_, ptr, row, val, control,            &
                      .FALSE., .TRUE. )
   END DO

!  computed scaling returned in scale, then reused as user scaling

   control%scaling = 1
   scale = zero
   CALL full_solve( 'scaling = 1 returns scale', 80_ip_, ptr, row, val,       &
                    control, .FALSE., .TRUE., scale = scale )
   CALL report( ALL( scale > zero ), 'returned Hungarian scaling positive' )
   control%scaling = 0
   CALL full_solve( 'scaling = 0 with user scale', 80_ip_, ptr, row, val,     &
                    control, .FALSE., .TRUE., scale = scale )

!  extreme (but exact power-of-2) user scaling

   scale = 2.0_rp_ ** ( MAXEXPONENT( one ) / 4 )
   CALL full_solve( 'user scale = 2**(maxexponent/4)', 80_ip_, ptr, row,      &
                    val, control, .FALSE., .TRUE., scale = scale )

!  auction parameters at non-default values

   control%scaling = 2
   control%auction%max_iterations = 1
   CALL full_solve( 'auction scaling, max_iterations = 1', 80_ip_, ptr, row,  &
                    val, control, .FALSE., .TRUE. )
   END SUBROUTINE case_scalings

   SUBROUTINE case_pivot_methods( )
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   INTEGER( ip_ ) :: pm, fpm, n
   CHARACTER( LEN = 60 ) :: lab
   CALL quiet( control )
   CALL kkt( 300_ip_, 200_ip_, ptr, row, val )
   n = 500
   control%nemin = 64                         ! bigger fronts, more pivoting
   DO pm = 1, 3
     DO fpm = 0, 3
       control%pivot_method = pm
       control%failed_pivot_method = fpm
       WRITE( lab, "( 'pivot_method = ', I0, ', failed_pivot_method = ', I0 )" )&
         pm, fpm
       CALL full_solve( TRIM( lab ), n, ptr, row, val, control, .FALSE.,      &
                        .TRUE., inform_out = inform )
       IF ( inform%flag >= 0 ) THEN
         IF ( inform%num_neg /= 200 ) CALL report( .FALSE.,                    &
           TRIM( lab ) // ' inertia' )
       END IF
     END DO
   END DO
   control%pivot_method = 0                   ! out of range values
   CALL full_solve( 'pivot_method = 0', n, ptr, row, val, control, .FALSE.,   &
                    .TRUE. )
   control%pivot_method = 99
   CALL full_solve( 'pivot_method = 99', n, ptr, row, val, control, .FALSE.,  &
                    .TRUE. )
   END SUBROUTINE case_pivot_methods

   SUBROUTINE case_threshold_u( )
   TYPE( SLBLT_control_type ) :: control
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : ), ptrd( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : ), rowd( : )
   REAL( rp_ ), ALLOCATABLE :: val( : ), vald( : )
   INTEGER( ip_ ) :: k, pm
   REAL( rp_ ), PARAMETER :: uu( 4 ) =                                         &
     (/ 0.0_rp_, 1.0E-10_rp_, 0.5_rp_, 1.0_rp_ /)
   CHARACTER( LEN = 60 ) :: lab
   CALL quiet( control )
   CALL kkt( 100_ip_, 60_ip_, ptr, row, val )
   CALL tridiag( 160_ip_, 4.0_rp_, one, .TRUE., ptrd, rowd, vald )
   control%nemin = 64
   DO pm = 1, 3, 2
     control%pivot_method = pm
     DO k = 1, SIZE( uu )
       control%u = uu( k )
       WRITE( lab, "( 'pivot_method = ', I0, ', u = ', ES8.1 )" ) pm, uu( k )
       IF ( uu( k ) < 1.0E-5_rp_ ) THEN

!  (almost) no pivoting (u <= 0 is replaced by EPSILON): accurate on a
!  diagonally dominant indefinite matrix; on the KKT matrix, at least the
!  solution must stay finite

         CALL full_solve( TRIM( lab ) // ', diag dominant', 160_ip_, ptrd,    &
                          rowd, vald, control, .FALSE., .TRUE. )
         CALL full_solve( TRIM( lab ) // ', KKT finite result', 160_ip_, ptr, &
                          row, val, control, .FALSE., .TRUE.,                  &
                          finite_only = .TRUE. )
       ELSE

!  moderate and large u (u > 0.5 is replaced by 0.5): expect an accurate
!  solution

         CALL full_solve( TRIM( lab ) // ', KKT', 160_ip_, ptr, row, val,     &
                          control, .FALSE., .TRUE. )
       END IF
     END DO
   END DO
   END SUBROUTINE case_threshold_u

!  the "emergency" 1x1 pivot of threshold partial pivoting, used when no
!  pivot passes the threshold test, on the 4x4 KKT matrix
!
!      [  4  -1   0   1 ]
!  A = [ -1   4  -1   1 ]
!      [  0  -1   4   0 ]
!      [  1   1   0   0 ]
!
!  with pivot_method = 3 (TPP), u = 0.5, nemin = 1. Each of the three
!  elimination orders below reaches the emergency pivot; its trailing update
!  was once applied one row too high, giving errors of about 0.4

   SUBROUTINE case_tpp_emergency_4x4( )
   INTEGER( ip_ ), PARAMETER :: n = 4
   INTEGER( i8_ ) :: ptr( n + 1 ) = (/ 1_i8_, 4_i8_, 7_i8_, 8_i8_, 9_i8_ /)
   INTEGER( ip_ ) :: row( 8 ) = (/ 1, 2, 4, 2, 3, 4, 3, 4 /)
   REAL( rp_ ) :: val( 8 ) = (/ 4.0_rp_, - 1.0_rp_, 1.0_rp_, 4.0_rp_,          &
                                - 1.0_rp_, 1.0_rp_, 4.0_rp_, 0.0_rp_ /)
   REAL( rp_ ), PARAMETER :: xs( n ) = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
   INTEGER( ip_ ), PARAMETER :: orders( n, 3 ) =                               &
     RESHAPE( (/ 1, 3, 4, 2,   1, 4, 2, 3,   2, 4, 1, 3 /), (/ n, 3_ip_ /) )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( ip_ ) :: order( n ), flag, k
   REAL( rp_ ) :: x( n ), b( n )
   CHARACTER( LEN = 60 ) :: lab
   CALL quiet( control )
   control%ordering = 0
   control%nemin = 1
   control%pivot_method = 3
   control%u = 0.5_rp_
   CALL symv( n, ptr, row, val, xs, b )
   DO k = 1, 3
     WRITE( lab, "( 'TPP emergency pivot 4x4, order ', 4I2 )" ) orders( :, k )
     order = orders( :, k )       ! SLBLT_analyse overwrites order
     CALL SLBLT_analyse( .TRUE., n, ptr, row, akeep, control, inform,          &
                         order = order )
     CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
     IF ( inform%flag /= SLBLT_SUCCESS ) THEN
       CALL report_flag( inform%flag, SLBLT_SUCCESS, TRIM( lab ) )
     ELSE
       x = b
       CALL SLBLT_solve( x, akeep, fkeep, control, inform )
       CALL report( inform%flag == 0 .AND. inform%num_neg == 1 .AND.           &
                    MAXVAL( ABS( x - xs ) ) <= tol, TRIM( lab ) )
     END IF
     CALL SLBLT_free( akeep, fkeep, flag )
   END DO
   END SUBROUTINE case_tpp_emergency_4x4

   SUBROUTINE case_small( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   INTEGER( ip_ ) :: flag
   CALL quiet( control )

!  diagonal matrix diag( 1, 1e-3, 1, 1e-3, ... ); with small = 1e-2 the tiny
!  pivots are treated as zero, so rank drops to n/2

   CALL tridiag( 10_ip_, one, zero, .FALSE., ptr, row, val )
   val( 3 : SIZE( val ) : 4 ) = 1.0E-3_rp_
   control%small = 1.0E-2_rp_
   control%action = .TRUE.
   CALL SLBLT_analyse( .TRUE., 10_ip_, ptr, row, akeep, control, inform )
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   CALL report( inform%flag == SLBLT_WARNING_FACT_SINGULAR .AND.               &
                inform%matrix_rank == 5, 'small = 1e-2 zeroes 1e-3 pivots' )
   control%small = HUGE( one )
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   CALL report( inform%flag == SLBLT_WARNING_FACT_SINGULAR .AND.               &
                inform%matrix_rank == 0, 'small = HUGE: all pivots zero' )
   control%small = - one                     ! negative: behaves like 0
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   CALL report( inform%flag == 0 .AND. inform%matrix_rank == 10,               &
                'small < 0: full rank' )
   CALL SLBLT_free( akeep, fkeep, flag )
   END SUBROUTINE case_small

   SUBROUTINE case_blocking_and_threads( )
   TYPE( SLBLT_control_type ) :: control
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   INTEGER( ip_ ) :: k, n
   INTEGER( ip_ ), PARAMETER :: bs( 5 ) = (/ - 5, 0, 1, 3, 100000 /)
   CHARACTER( LEN = 60 ) :: lab
   CALL quiet( control )

!  tiny block sizes are very slow, so use a smaller matrix for these

   CALL kkt( 30_ip_, 20_ip_, ptr, row, val )
   n = 50
   control%nemin = 128
   DO k = 1, SIZE( bs )
     control%block_size = bs( k )
     WRITE( lab, "( 'block_size = ', I0 )" ) bs( k )
     CALL full_solve( TRIM( lab ), n, ptr, row, val, control, .FALSE., .TRUE. )
     control%pivot_method = 1
     CALL full_solve( TRIM( lab ) // ', aggressive APP', n, ptr, row, val,    &
                      control, .FALSE., .TRUE. )
     control%pivot_method = 2
   END DO
   control%block_size = 256
   CALL kkt( 300_ip_, 200_ip_, ptr, row, val )
   n = 500
   control%small_subtree_threshold = 0
   CALL full_solve( 'small_subtree_threshold = 0', n, ptr, row, val,          &
                    control, .FALSE., .TRUE. )
   control%small_subtree_threshold = HUGE( 1_long_ )
   CALL full_solve( 'small_subtree_threshold = HUGE', n, ptr, row, val,       &
                    control, .FALSE., .TRUE. )
   control%small_subtree_threshold = 4 * 10 ** 6
   control%multiplier = 1.0_rp_
   CALL full_solve( 'multiplier = 1.0', n, ptr, row, val, control, .FALSE.,   &
                    .TRUE. )
   control%multiplier = 10.0_rp_
   CALL full_solve( 'multiplier = 10.0', n, ptr, row, val, control, .FALSE.,  &
                    .TRUE. )
   END SUBROUTINE case_blocking_and_threads

   SUBROUTINE case_printing( )
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   INTEGER( ip_ ) :: flag, lines, ios, lv
   INTEGER( ip_ ), PARAMETER :: out = 47
   INTEGER( ip_ ), PARAMETER :: levels( 3 ) = (/ 0_ip_, 1_ip_, 3_ip_ /)
   CHARACTER( LEN = 200 ) :: line
   CHARACTER( LEN = 60 ) :: lab
   CALL tridiag( 30_ip_, 4.0_rp_, one, .TRUE., ptr, row, val )

!  verbose printing to a scratch unit, including on an error path

   DO lv = 1, SIZE( levels )
     OPEN( out, STATUS = 'SCRATCH' )
     control%print_level = levels( lv )
     control%unit_diagnostics = out
     control%unit_error = out
     control%unit_warning = out
     WRITE( lab, "( 'print_level = ', I0, ' to scratch unit' )" ) levels( lv )
     CALL full_solve( TRIM( lab ), 30_ip_, ptr, row, val, control, .FALSE.,   &
                      .TRUE. )
     control%ordering = 7                     ! provoke an error message
     CALL SLBLT_analyse( .TRUE., 30_ip_, ptr, row, akeep, control, inform )
     control%ordering = 1
     REWIND( out )
     lines = 0
     DO
       READ( out, "( A )", IOSTAT = ios ) line
       IF ( ios /= 0 ) EXIT
       lines = lines + 1
     END DO
     CLOSE( out )
     CALL report( lines > 0, TRIM( lab ) // ' produced output' )
   END DO

!  print_level = -1: nothing at all may be written, even for errors

   OPEN( out, STATUS = 'SCRATCH' )
   control%print_level = - 1
   control%ordering = 7
   CALL SLBLT_analyse( .TRUE., 30_ip_, ptr, row, akeep, control, inform )
   control%ordering = 1
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   REWIND( out )
   READ( out, "( A )", IOSTAT = ios ) line
   CLOSE( out )
   CALL report( ios /= 0, 'print_level = -1 is silent on errors' )

!  negative units with high print level must not write anything / crash

   control%print_level = 3
   control%unit_diagnostics = - 1
   control%unit_error = - 1
   control%unit_warning = - 1
   CALL full_solve( 'print_level = 3 with negative units', 30_ip_, ptr, row,  &
                    val, control, .FALSE., .TRUE. )
   CALL SLBLT_free( akeep, fkeep, flag )
   END SUBROUTINE case_printing

!  reuse of akeep/fkeep: analyse once, factor several different value sets;
!  re-analyse into a used akeep

   SUBROUTINE case_refactor_reuse( )
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : ), v( : ), x( : ), b( : ), xs( : )
   INTEGER( ip_ ) :: flag, it, n
   LOGICAL :: ok
   CALL quiet( control )
   CALL kkt( 40_ip_, 25_ip_, ptr, row, val )
   n = 65
   ALLOCATE( v( SIZE( val ) ), x( n ), b( n ), xs( n ) )
   CALL make_solution( n, xs )
   CALL SLBLT_analyse( .TRUE., n, ptr, row, akeep, control, inform )
   ok = inform%flag >= 0
   DO it = 1, 5
     v = val * REAL( it, rp_ )
     IF ( MOD( it, 2 ) == 0 ) v = - v
     CALL SLBLT_factor( .FALSE., v, akeep, fkeep, control, inform )
     ok = ok .AND. inform%flag >= 0
     CALL symv( n, ptr, row, v, xs, b )
     x = b
     CALL SLBLT_solve( x, akeep, fkeep, control, inform )
     ok = ok .AND. inform%flag >= 0 .AND.                                      &
          backward_error( n, ptr, row, v, x, b ) <= tol
   END DO
   CALL report( ok, 'refactorize 5 times with one analyse' )

!  re-analyse into the used akeep with a different matrix and size

   CALL tridiag( 17_ip_, 4.0_rp_, one, .TRUE., ptr, row, val )
   DEALLOCATE( x, b, xs ) ; ALLOCATE( x( 17 ), b( 17 ), xs( 17 ) )
   CALL SLBLT_analyse( .TRUE., 17_ip_, ptr, row, akeep, control, inform )
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   CALL make_solution( 17_ip_, xs )
   CALL symv( 17_ip_, ptr, row, val, xs, b )
   x = b
   CALL SLBLT_solve( x, akeep, fkeep, control, inform )
   CALL report( inform%flag >= 0 .AND.                                         &
                backward_error( 17_ip_, ptr, row, val, x, b ) <= tol,          &
                're-analyse used akeep with new matrix' )

!  switch the same fkeep between posdef and indefinite

   CALL tridiag( 17_ip_, 4.0_rp_, - one, .FALSE., ptr, row, val )
   CALL SLBLT_analyse( .TRUE., 17_ip_, ptr, row, akeep, control, inform )
   CALL SLBLT_factor( .TRUE., val, akeep, fkeep, control, inform )
   CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
   CALL symv( 17_ip_, ptr, row, val, xs, b )
   x = b
   CALL SLBLT_solve( x, akeep, fkeep, control, inform, job = 2_ip_ )
   CALL report( inform%flag == 0, 'posdef then indefinite factor in one fkeep' )
   CALL SLBLT_free( akeep, fkeep, flag )
   CALL report_flag( flag, 0_ip_, 'final free' )
   END SUBROUTINE case_refactor_reuse

!  minimal example of a delayed pivot crossing a subtree boundary:
!
!            x1   z    r
!      x1 [   4   0   -1 ]
!  A = z  [   0   0    1 ]      det( A ) = -4 (nonsingular)
!      r  [  -1   1    4 ]
!
!  z has a zero diagonal and is coupled only to r, so it must be delayed to
!  r's node and eliminated with r as a 2x2 pivot. With the natural order,
!  nemin = 1 and max_load_inbalance = 1.0 the tree is split so that z and r
!  lie in different subtrees, and the delay has to be passed from the child
!  subtree to its parent. This used to fail with SLBLT_ERROR_SINGULAR

   SUBROUTINE case_subtree_delay_3x3( )
   INTEGER( ip_ ), PARAMETER :: n = 3
   INTEGER( i8_ ) :: ptr( n + 1 ) = (/ 1_i8_, 3_i8_, 5_i8_, 6_i8_ /)
   INTEGER( ip_ ) :: row( 5 ) = (/ 1, 3, 2, 3, 3 /)
   REAL( rp_ ) :: val( 5 ) =                                                   &
     (/ 4.0_rp_, - 1.0_rp_, 0.0_rp_, 1.0_rp_, 4.0_rp_ /)
   REAL( rp_ ), PARAMETER :: xs( n ) = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_ /)
   REAL( rp_ ), PARAMETER :: imbalance( 2 ) = (/ 1.0_rp_, 1.2_rp_ /)
   TYPE( SLBLT_akeep_type ) :: akeep
   TYPE( SLBLT_fkeep_type ) :: fkeep
   TYPE( SLBLT_control_type ) :: control
   TYPE( SLBLT_inform_type ) :: inform
   INTEGER( ip_ ) :: order( n ), flag, k, nparts
   REAL( rp_ ) :: x( n ), b( n )
   CHARACTER( LEN = 60 ) :: lab
   CALL quiet( control )
   control%ordering = 0
   control%nemin = 1
   CALL symv( n, ptr, row, val, xs, b )
   DO k = 1, 2
     control%max_load_inbalance = REAL( imbalance( k ) )
     WRITE( lab, "( '3x3 delay example, max_load_inbalance = ', F3.1 )" )      &
       imbalance( k )

!  SLBLT_analyse overwrites order with the pivot order used, so reset it

     order = (/ 1, 2, 3 /)
     CALL SLBLT_analyse( .TRUE., n, ptr, row, akeep, control, inform,          &
                         order = order )
     nparts = inform%nparts

!  the split case only tests the cross-subtree path if a split happens

     IF ( k == 1 ) CALL report( nparts == 2, TRIM( lab ) // ': 2 subtrees' )
     CALL SLBLT_factor( .FALSE., val, akeep, fkeep, control, inform )
     CALL report_flag( inform%flag, SLBLT_SUCCESS, TRIM( lab ) // ': factor' )
     CALL report( inform%num_delay == 1 .AND. inform%num_neg == 1 .AND.        &
                  inform%num_two == 1 .AND. inform%matrix_rank == n,           &
                  TRIM( lab ) // ': delays, inertia, rank' )
     x = b
     CALL SLBLT_solve( x, akeep, fkeep, control, inform )
     CALL report( inform%flag == 0 .AND. MAXVAL( ABS( x - xs ) ) <= tol,       &
                  TRIM( lab ) // ': solution (1, 2, 3)' )
     CALL SLBLT_free( akeep, fkeep, flag )
   END DO
   END SUBROUTINE case_subtree_delay_3x3

!  subtree partitioning controls (NUMA handling and load imbalance)

   SUBROUTINE case_subtree_partition( )
   TYPE( SLBLT_control_type ) :: control
   INTEGER( i8_ ), ALLOCATABLE :: ptr( : )
   INTEGER( ip_ ), ALLOCATABLE :: row( : )
   REAL( rp_ ), ALLOCATABLE :: val( : )
   INTEGER( ip_ ) :: n
   CALL quiet( control )
   CALL kkt( 300_ip_, 200_ip_, ptr, row, val )
   n = 500
   control%nemin = 128
   control%max_load_inbalance = HUGE( 1.0 )
   CALL full_solve( 'max_load_inbalance = HUGE', n, ptr, row, val, control,   &
                    .FALSE., .TRUE. )
   control%max_load_inbalance = 1.2
   control%ignore_numa = .FALSE.
   CALL full_solve( 'ignore_numa = .false.', n, ptr, row, val, control,       &
                    .FALSE., .TRUE. )
   control%ignore_numa = .TRUE.

!  SPD matrix split into several subtrees (no delayed pivots)

   CALL tridiag( n, 4.0_rp_, - one, .FALSE., ptr, row, val )
   control%max_load_inbalance = 1.0
   CALL full_solve( 'max_load_inbalance = 1.0, SPD', n, ptr, row, val,        &
                    control, .TRUE., .TRUE. )

!  KKT matrix split into several subtrees (delays cross subtree boundaries)

   CALL kkt( 300_ip_, 200_ip_, ptr, row, val )
   CALL full_solve( 'max_load_inbalance = 1.0, KKT', n, ptr, row, val,        &
                    control, .FALSE., .TRUE. )
   END SUBROUTINE case_subtree_partition

   END PROGRAM GALAHAD_SLBLT_edge_test_program
