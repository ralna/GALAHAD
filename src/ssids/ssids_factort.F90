! THIS VERSION: GALAHAD 5.5 - 2026-07-27
!
! Unit tests for the pure-Fortran SSIDS kernels (GALAHAD_SSIDS_factor_precision),
! templated over precision/integer via the usual _precision mechanism. Two levels:
!   * per-routine kernel tests: calc_ld, ldlt_tpp_factor, ldlt_blocked_factor,
!     factor_node_indef, assemble_expected(_contrib), ldlt_app_solve_{fwd,diag,bwd};
!   * driver/integration tests: 2-node SPD, 3-level (post-assembly), 2x2-pivot
!     indefinite, forced delayed pivots, blocked path, foreign child_contrib.

#include "galahad_modules.h"

   PROGRAM GALAHAD_SSIDS_factor_test_program
   USE GALAHAD_KINDS_precision, ONLY : ip_, rp_
   USE GALAHAD_SSIDS_factor_precision, ONLY : dmf_node,               &
        subtree_contrib_t, factor_subtree_delay, extract_contrib,            &
        subtree_solve_fwd_delay, subtree_solve_diag_delay,                   &
        subtree_solve_bwd_delay,                                             &
        calc_ld, ldlt_tpp_factor, ldlt_blocked_factor, factor_node_indef,   &
        block_ldlt, ldlt_app_factor,                                        &
        assemble_expected, assemble_expected_contrib,                       &
        ldlt_app_solve_fwd, ldlt_app_solve_diag, ldlt_app_solve_bwd
   IMPLICIT NONE
   INTEGER( ip_ ) :: nfail
   nfail = 0

   ! per-routine kernel tests
   CALL case_calc_ld( nfail )
   CALL case_ldlt_tpp( nfail )
   CALL case_ldlt_blocked( nfail )
   CALL case_block_ldlt( nfail )
   CALL case_ldlt_app( nfail )
   CALL case_ldlt_app_rec( nfail )
   CALL case_ldlt_app_aggr( nfail )
   CALL case_singular_action( nfail )
   CALL case_factor_node( nfail )
   CALL case_assemble( nfail )
   CALL case_app_solve( nfail )
   ! driver / integration tests
   CALL case_posdef( nfail )
   CALL case_2node_spd( nfail )
   CALL case_3level_spd( nfail )
   CALL case_indef_2x2( nfail )
   CALL case_delay( nfail )
   CALL case_blocked( nfail )
   CALL case_child_contrib( nfail )

   IF ( nfail == 0 ) THEN
      WRITE( 6, "( ' ssids fortran kernels: all tests passed' )" )
   ELSE
      WRITE( 6, "( ' ssids fortran kernels: ', I0, ' FAILED' )" ) nfail
      STOP 1
   END IF

 CONTAINS

   ! ---- helpers ----
   REAL( rp_ ) FUNCTION tol( A )
      REAL( rp_ ), INTENT( IN ) :: A(:,:)
      tol = SQRT( EPSILON( 1.0_rp_ ) ) * MAX( MAXVAL( ABS( A ) ), 1.0_rp_ ) * 100.0_rp_
   END FUNCTION tol

   SUBROUTINE set_node( nd, ncol, rlist, parent )
      TYPE( dmf_node ), INTENT( OUT ) :: nd
      INTEGER( ip_ ),   INTENT( IN )  :: ncol, rlist(:), parent
      nd%symb_ncol = ncol; nd%symb_nrow = SIZE( rlist ); nd%parent = parent
      ALLOCATE( nd%rlist( nd%symb_nrow ) ); nd%rlist = rlist
   END SUBROUTINE set_node

   SUBROUTINE set_a( nd, ai, aj, av )
      TYPE( dmf_node ), INTENT( INOUT ) :: nd
      INTEGER( ip_ ),   INTENT( IN ) :: ai(:), aj(:)
      REAL( rp_ ),      INTENT( IN ) :: av(:)
      nd%ai = ai; nd%aj = aj; nd%av = av
   END SUBROUTINE set_a

   SUBROUTINE solve_check( node, nn, A, b, n, nb, label, nfail, posdef )
      TYPE( dmf_node ), INTENT( INOUT ) :: node(:)
      INTEGER( ip_ ),   INTENT( IN )    :: nn, n, nb
      REAL( rp_ ),      INTENT( IN )    :: A(n,n), b(n)
      CHARACTER( * ),   INTENT( IN )    :: label
      INTEGER( ip_ ),   INTENT( INOUT ) :: nfail
      LOGICAL, OPTIONAL, INTENT( IN )   :: posdef
      REAL( rp_ ) :: x(n,1), err
      LOGICAL :: ok, pd
      pd = .FALSE. ; IF ( PRESENT( posdef ) ) pd = posdef
      CALL factor_subtree_delay( node, nn, n, .TRUE., 0.01_rp_,               &
                                 EPSILON( 1.0_rp_ ), nb, pd, ok )
      x( :, 1 ) = b
      CALL subtree_solve_fwd_delay ( node, nn, 1_ip_, x, n )
      CALL subtree_solve_diag_delay( node, nn, 1_ip_, x, n )
      CALL subtree_solve_bwd_delay ( node, nn, 1_ip_, x, n )
      err = MAXVAL( ABS( MATMUL( A, x( :, 1 ) ) - b ) )
      IF ( ok .AND. err <= tol( A ) ) THEN
         WRITE( 6, "( '  ok   : ', A )" ) label
      ELSE
         WRITE( 6, "( '  FAIL : ', A, '  err=', ES10.3 )" ) label, err
         nfail = nfail + 1
      END IF
   END SUBROUTINE solve_check

   SUBROUTINE report( cond, label, nfail )
      LOGICAL,        INTENT( IN )    :: cond
      CHARACTER( * ), INTENT( IN )    :: label
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      IF ( cond ) THEN
         WRITE( 6, "( '  ok   : ', A )" ) label
      ELSE
         WRITE( 6, "( '  FAIL : ', A )" ) label
         nfail = nfail + 1
      END IF
   END SUBROUTINE report

   !> Solve A x = b from a completed dense LDL^T factor (L in fac, perm, d),
   !! using the ldlt_app fwd/diag/bwd kernels.
   SUBROUTINE dense_solve( n, fac, perm, d, b, x )
      INTEGER( ip_ ), INTENT( IN )  :: n
      REAL( rp_ ),    INTENT( IN )  :: fac(n,n), d(*)
      INTEGER( ip_ ), INTENT( IN )  :: perm(n)
      REAL( rp_ ),    INTENT( IN )  :: b(n)
      REAL( rp_ ),    INTENT( OUT ) :: x(n)
      REAL( rp_ ) :: bp(n)
      INTEGER( ip_ ) :: i
      DO i = 1, n
         bp(i) = b( perm(i) )
      END DO
      CALL ldlt_app_solve_fwd ( n, n, fac, n, 1_ip_, bp, n )
      CALL ldlt_app_solve_diag( n, d, 1_ip_, bp, n )
      CALL ldlt_app_solve_bwd ( n, n, fac, n, 1_ip_, bp, n )
      DO i = 1, n
         x( perm(i) ) = bp(i)
      END DO
   END SUBROUTINE dense_solve

   ! ---- per-routine kernel tests ----

   SUBROUTINE case_calc_ld( nfail )
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      INTEGER( ip_ ), PARAMETER :: m = 4, n = 2
      REAL( rp_ ) :: L(m,n), d(2*n+2), LD(m,n), err, dc
      INTEGER( ip_ ) :: r, c
      DO c = 1, n ; DO r = 1, m ; L(r,c) = 0.5_rp_*r + c ; END DO ; END DO
      d = 0.0_rp_
      d(1) = 2.0_rp_        ! D^-1 for col 1 (=> D = 0.5); also finite 2x2-marker for col 0
      d(3) = 4.0_rp_        ! D^-1 for col 2 (=> D = 0.25)
      CALL calc_ld( .FALSE., m, n, L, m, d, LD, m )
      err = 0.0_rp_
      DO c = 1, n
         dc = 1.0_rp_ / d( 2*(c-1)+1 )
         DO r = 1, m
            err = MAX( err, ABS( LD(r,c) - dc*L(r,c) ) )
         END DO
      END DO
      CALL report( err <= SQRT( EPSILON( 1.0_rp_ ) ), "calc_ld (LD = L*D)", nfail )
   END SUBROUTINE case_calc_ld

   SUBROUTINE spd_matrix( n, A )
      INTEGER( ip_ ), INTENT( IN )  :: n
      REAL( rp_ ),    INTENT( OUT ) :: A(n,n)
      INTEGER( ip_ ) :: i, j
      DO j = 1, n ; DO i = 1, n
         A(i,j) = 1.0_rp_ / ( 1.0_rp_ + ABS( i - j ) )
      END DO ; END DO
      DO i = 1, n
         A(i,i) = A(i,i) + REAL( n, rp_ )
      END DO
   END SUBROUTINE spd_matrix

   SUBROUTINE case_ldlt_tpp( nfail )
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      INTEGER( ip_ ), PARAMETER :: n = 8
      REAL( rp_ ) :: A(n,n), fac(n,n), d(2*n+2), ldw(n,2), aleft(1,1), b(n), x(n)
      INTEGER( ip_ ) :: perm(n), flag, nelim, i
      CALL spd_matrix( n, A )
      fac = A ; DO i = 1, n ; perm(i) = i ; END DO
      nelim = ldlt_tpp_factor( n, n, perm, fac, n, d, ldw, n, .TRUE., 0.01_rp_, &
                               EPSILON( 1.0_rp_ ), 0_ip_, aleft, 1_ip_, flag )
      DO i = 1, n ; b(i) = REAL( i, rp_ ) ; END DO
      CALL dense_solve( n, fac, perm, d, b, x )
      CALL report( nelim == n .AND. MAXVAL( ABS( MATMUL(A,x) - b ) ) <= tol(A), &
                   "ldlt_tpp_factor + solve", nfail )
   END SUBROUTINE case_ldlt_tpp

   SUBROUTINE case_ldlt_blocked( nfail )
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      INTEGER( ip_ ), PARAMETER :: n = 10
      REAL( rp_ ) :: A(n,n), fac(n,n), d(2*n+2), b(n), x(n)
      INTEGER( ip_ ) :: perm(n), flag, nelim, i
      CALL spd_matrix( n, A )
      fac = A ; DO i = 1, n ; perm(i) = i ; END DO
      nelim = ldlt_blocked_factor( n, n, perm, fac, n, d, 0.01_rp_,             &
                                   EPSILON( 1.0_rp_ ), .TRUE., 3_ip_, flag )
      DO i = 1, n ; b(i) = REAL( n - i + 1, rp_ ) ; END DO
      CALL dense_solve( n, fac, perm, d, b, x )
      CALL report( nelim == n .AND. MAXVAL( ABS( MATMUL(A,x) - b ) ) <= tol(A), &
                   "ldlt_blocked_factor (nb=3) + solve", nfail )
   END SUBROUTINE case_ldlt_blocked

   !> block_ldlt (Bunch-Kaufman) on a full block; matrix designed to force a
   !! 2x2 pivot (small diagonal, large off-diagonal in the leading 2x2).
   SUBROUTINE case_block_ldlt( nfail )
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      INTEGER( ip_ ), PARAMETER :: n = 4
      REAL( rp_ ) :: A(n,n), fac(n,n), d(2*n+2), ldw(n,n), b(n), x(n)
      INTEGER( ip_ ) :: perm(n), lperm(n), flag, i
      A = RESHAPE( [ 0.1_rp_,  3.0_rp_, 0.0_rp_,  0.0_rp_,                      &
                     3.0_rp_,  0.1_rp_, 0.0_rp_,  0.0_rp_,                      &
                     0.0_rp_,  0.0_rp_, 5.0_rp_,  1.0_rp_,                      &
                     0.0_rp_,  0.0_rp_, 1.0_rp_, -5.0_rp_ ], [n,n] )
      fac = A ; DO i = 1, n ; perm(i) = i ; lperm(i) = i-1 ; END DO
      CALL block_ldlt( 0_ip_, perm, fac, n, d, ldw, .TRUE., 0.01_rp_,          &
                       EPSILON( 1.0_rp_ ), lperm, n, flag )
      DO i = 1, n ; b(i) = REAL( i, rp_ ) ; END DO
      CALL dense_solve( n, fac, perm, d, b, x )
      CALL report( flag == 0 .AND. MAXVAL( ABS( MATMUL(A,x) - b ) ) <= tol(A), &
                   "block_ldlt (Bunch-Kaufman) + solve", nfail )
   END SUBROUTINE case_block_ldlt

   !> singular front: with action=.FALSE. the a-posteriori pivoted factor must
   !! ABORT (flag /= 0), not silently report a full elimination; with
   !! action=.TRUE. it must NOT spuriously abort (null pivots are zeroed/delayed).
   !! Guards the pivoted-pass singular detection (block_ldlt / ldlt_tpp flag).
   SUBROUTINE case_singular_action( nfail )
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      INTEGER( ip_ ), PARAMETER :: n = 8
      REAL( rp_ ) :: A(n,n), fac(n,n), d(2*n+2), cdum(1,1)
      INTEGER( ip_ ) :: perm(n), flag, nelim, ndout, i, j
      DO j = 1, n ; DO i = 1, n ; A(i,j) = 1.0_rp_ ; END DO ; END DO  ! rank-1
      fac = A ; DO i = 1, n ; perm(i) = i ; END DO
      nelim = ldlt_app_factor( n, n, perm, fac, n, d, 0.01_rp_,                 &
                               EPSILON( 1.0_rp_ ), .FALSE., 4_ip_, flag )
      CALL report( flag /= 0,                                                   &
                   "ldlt_app_factor singular, action=.FALSE. -> abort", nfail )
      fac = A ; DO i = 1, n ; perm(i) = i ; END DO
      nelim = ldlt_app_factor( n, n, perm, fac, n, d, 0.01_rp_,                 &
                               EPSILON( 1.0_rp_ ), .TRUE., 4_ip_, flag )
      CALL report( flag == 0,                                                   &
                   "ldlt_app_factor singular, action=.TRUE. -> no abort", nfail )
      ! factor_node_indef must propagate the abort as nelim < 0 (-> ERROR_SINGULAR)
      fac = A ; DO i = 1, n ; perm(i) = i ; END DO
      CALL factor_node_indef( n, n, 0_ip_, fac, n, d, perm, cdum, 1_ip_, .FALSE.,&
                              0.01_rp_, EPSILON( 1.0_rp_ ), n, .FALSE., nelim,   &
                              ndout )
      CALL report( nelim < 0,                                                   &
                   "factor_node_indef singular, action=.FALSE. -> nelim<0", nfail )
   END SUBROUTINE case_singular_action

   !> ldlt_app_factor: a-posteriori pivoted multi-block factor (nb=4 => 3 blocks)
   !! of an indefinite, diagonally dominant system.
   SUBROUTINE case_ldlt_app( nfail )
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      INTEGER( ip_ ), PARAMETER :: n = 12
      REAL( rp_ ) :: A(n,n), fac(n,n), d(2*n+2), b(n), x(n)
      INTEGER( ip_ ) :: perm(n), flag, nelim, i, j
      DO j = 1, n ; DO i = 1, n
         A(i,j) = 1.0_rp_ / ( 1.0_rp_ + ABS( i - j ) )
      END DO ; END DO
      DO i = 1, n           ! alternating-sign large diagonal => indefinite
         A(i,i) = MERGE( 6.0_rp_, -6.0_rp_, MOD( i, 2 ) == 0 )
      END DO
      fac = A ; DO i = 1, n ; perm(i) = i ; END DO
      nelim = ldlt_app_factor( n, n, perm, fac, n, d, 0.01_rp_,                &
                               EPSILON( 1.0_rp_ ), .TRUE., 4_ip_, flag )
      DO i = 1, n ; b(i) = REAL( n - i + 1, rp_ ) ; END DO
      CALL dense_solve( n, fac, perm, d, b, x )
      CALL report( nelim == n .AND. MAXVAL( ABS( MATMUL(A,x) - b ) ) <= tol(A), &
                   "ldlt_app_factor (APP, nb=4) + solve", nfail )
   END SUBROUTINE case_ldlt_app

   !> ldlt_app_factor with an outer block size > 32: exercises the recursive
   !! inner-block (block_ldlt) factorization of the wide diagonal blocks.
   SUBROUTINE case_ldlt_app_rec( nfail )
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      INTEGER( ip_ ), PARAMETER :: n = 100
      REAL( rp_ ) :: A(n,n), fac(n,n), d(2*n+2), b(n), x(n)
      INTEGER( ip_ ) :: perm(n), flag, nelim, i, j
      DO j = 1, n ; DO i = 1, n
         A(i,j) = 1.0_rp_ / ( 1.0_rp_ + ABS( i - j ) )
      END DO ; END DO
      DO i = 1, n
         A(i,i) = MERGE( 9.0_rp_, -9.0_rp_, MOD( i, 2 ) == 0 )
      END DO
      fac = A ; DO i = 1, n ; perm(i) = i ; END DO
      ! nb=40 > 32 => outer 40-blocks, inner 32-blocks via recursion
      nelim = ldlt_app_factor( n, n, perm, fac, n, d, 0.01_rp_,                &
                               EPSILON( 1.0_rp_ ), .TRUE., 40_ip_, flag )
      DO i = 1, n ; b(i) = 1.0_rp_ + MOD( REAL( i, rp_ ), 5.0_rp_ ) ; END DO
      CALL dense_solve( n, fac, perm, d, b, x )
      CALL report( nelim == n .AND. MAXVAL( ABS( MATMUL(A,x) - b ) ) <= tol(A), &
                   "ldlt_app_factor (APP recursion, nb=40) + solve", nfail )
   END SUBROUTINE case_ldlt_app_rec

   !> Aggressive (unpivoted-first) path: on an SPD system it must match the
   !! pivoted result bit-for-bit; on an indefinite one it falls back and still
   !! yields a correct solve.
   SUBROUTINE case_ldlt_app_aggr( nfail )
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      INTEGER( ip_ ), PARAMETER :: n = 80
      REAL( rp_ ) :: A(n,n), fa(n,n), fp(n,n), da(2*n+2), dp(2*n+2), b(n), x(n)
      INTEGER( ip_ ) :: pa(n), pp(n), na, np, flag, i, j
      LOGICAL :: okspd, okindef
      ! (a) SPD: aggressive == pivoted bit-for-bit
      CALL spd_matrix( n, A )
      fa = A ; fp = A ; DO i = 1, n ; pa(i) = i ; pp(i) = i ; END DO
      na = ldlt_app_factor( n, n, pa, fa, n, da, 0.01_rp_, EPSILON(1.0_rp_),    &
                            .TRUE., 40_ip_, flag, aggressive = .TRUE. )
      np = ldlt_app_factor( n, n, pp, fp, n, dp, 0.01_rp_, EPSILON(1.0_rp_),    &
                            .TRUE., 40_ip_, flag )
      okspd = ( na == np ) .AND. ( MAXVAL(ABS(fa-fp)) == 0.0_rp_ ) .AND.        &
              ( MAXVAL(ABS(da(1:2*n)-dp(1:2*n))) == 0.0_rp_ ) .AND.             &
              ALL( pa == pp )
      ! (b) indefinite: aggressive falls back, solve still correct
      DO j = 1, n ; DO i = 1, n
         A(i,j) = 1.0_rp_ / ( 1.0_rp_ + ABS( i - j ) )
      END DO ; END DO
      DO i = 1, n ; A(i,i) = MERGE( 5.0_rp_, -5.0_rp_, MOD(i,2)==0 ) ; END DO
      fa = A ; DO i = 1, n ; pa(i) = i ; END DO
      na = ldlt_app_factor( n, n, pa, fa, n, da, 0.01_rp_, EPSILON(1.0_rp_),    &
                            .TRUE., 40_ip_, flag, aggressive = .TRUE. )
      DO i = 1, n ; b(i) = 1.0_rp_ + MOD( REAL(i,rp_), 4.0_rp_ ) ; END DO
      CALL dense_solve( n, fa, pa, da, b, x )
      okindef = ( na == n ) .AND. ( MAXVAL(ABS(MATMUL(A,x)-b)) <= tol(A) )
      CALL report( okspd, "ldlt_app aggressive == pivoted (SPD)", nfail )
      CALL report( okindef, "ldlt_app aggressive fallback (indef) + solve", nfail )
   END SUBROUTINE case_ldlt_app_aggr

   SUBROUTINE case_factor_node( nfail )
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      INTEGER( ip_ ), PARAMETER :: n = 6
      REAL( rp_ ) :: A(n,n), fac(n,n), d(2*n+2), cdum(1,1), b(n), x(n)
      INTEGER( ip_ ) :: perm(n), nelim, ndout, i
      CALL spd_matrix( n, A )
      fac = A ; DO i = 1, n ; perm(i) = i ; END DO
      ! square node (nrow=ncol=n): full elimination, no contribution block
      CALL factor_node_indef( n, n, 0_ip_, fac, n, d, perm, cdum, 1_ip_, .TRUE., &
                              0.01_rp_, EPSILON( 1.0_rp_ ), n, .FALSE., nelim, ndout )
      DO i = 1, n ; b(i) = 1.0_rp_ ; END DO
      CALL dense_solve( n, fac, perm, d, b, x )
      CALL report( nelim == n .AND. ndout == 0 .AND.                            &
                   MAXVAL( ABS( MATMUL(A,x) - b ) ) <= tol(A),                  &
                   "factor_node_indef (square) + solve", nfail )
   END SUBROUTINE case_factor_node

   SUBROUTINE case_assemble( nfail )
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      REAL( rp_ ) :: contrib(2,2), lcol(3,3), pcontrib(2,2)
      INTEGER( ip_ ) :: cache(2)
      LOGICAL :: ok1, ok2
      ! assemble_expected: child 2x2 contrib scattered into parent lcol cols
      contrib = RESHAPE( [ 1._rp_, 2._rp_, 2._rp_, 3._rp_ ], [2,2] )
      lcol = 0.0_rp_ ; cache = [ 1_ip_, 2_ip_ ]
      CALL assemble_expected( 1_ip_, 2_ip_, 2_ip_, cache, contrib, 2_ip_, lcol, &
                              3_ip_, 3_ip_ )
      ok1 = ( ABS( lcol(1,1)-1._rp_ ) + ABS( lcol(2,1)-2._rp_ ) +               &
              ABS( lcol(2,2)-3._rp_ ) ) <= SQRT( EPSILON( 1.0_rp_ ) )
      ! assemble_expected_contrib: into a contribution block (cache >= 1)
      pcontrib = 0.0_rp_ ; cache = [ 1_ip_, 2_ip_ ]
      CALL assemble_expected_contrib( 1_ip_, 2_ip_, 2_ip_, cache, contrib,      &
                                      2_ip_, pcontrib, 2_ip_ )
      ok2 = ( ABS( pcontrib(1,1)-1._rp_ ) + ABS( pcontrib(2,1)-2._rp_ ) +       &
              ABS( pcontrib(2,2)-3._rp_ ) ) <= SQRT( EPSILON( 1.0_rp_ ) )
      CALL report( ok1, "assemble_expected (scatter into lcol)", nfail )
      CALL report( ok2, "assemble_expected_contrib (scatter into contrib)", nfail )
   END SUBROUTINE case_assemble

   SUBROUTINE case_app_solve( nfail )
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      INTEGER( ip_ ), PARAMETER :: n = 6, nrhs = 2
      REAL( rp_ ) :: A(n,n), fac(n,n), d(2*n+2), ldw(n,2), aleft(1,1)
      REAL( rp_ ) :: B(n,nrhs), X(n,nrhs), bp(n,nrhs)
      INTEGER( ip_ ) :: perm(n), flag, nelim, i, r
      ! exercise the multi-RHS fwd/diag/bwd solve path
      CALL spd_matrix( n, A )
      fac = A ; DO i = 1, n ; perm(i) = i ; END DO
      nelim = ldlt_tpp_factor( n, n, perm, fac, n, d, ldw, n, .TRUE., 0.01_rp_, &
                               EPSILON( 1.0_rp_ ), 0_ip_, aleft, 1_ip_, flag )
      DO i = 1, n ; B(i,1) = REAL( i, rp_ ) ; B(i,2) = 1.0_rp_ ; END DO
      DO r = 1, nrhs ; DO i = 1, n ; bp(i,r) = B( perm(i), r ) ; END DO ; END DO
      CALL ldlt_app_solve_fwd ( n, n, fac, n, nrhs, bp, n )
      CALL ldlt_app_solve_diag( n, d, nrhs, bp, n )
      CALL ldlt_app_solve_bwd ( n, n, fac, n, nrhs, bp, n )
      DO r = 1, nrhs ; DO i = 1, n ; X( perm(i), r ) = bp(i,r) ; END DO ; END DO
      CALL report( MAXVAL( ABS( MATMUL(A,X) - B ) ) <= tol(A),                  &
                   "ldlt_app_solve fwd/diag/bwd (multi-RHS)", nfail )
   END SUBROUTINE case_app_solve

   ! ---- driver / integration cases ----
   SUBROUTINE case_posdef( nfail )
      ! same 2-node SPD tree, factored via the Cholesky (posdef) path
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      INTEGER( ip_ ), PARAMETER :: n = 5
      REAL( rp_ ) :: A(n,n), b(n)
      TYPE( dmf_node ) :: node(2)
      A = RESHAPE( [ 10._rp_, 1._rp_, 2._rp_, 1._rp_, 0._rp_,                   &
                     1._rp_,10._rp_, 1._rp_, 2._rp_, 0._rp_,                    &
                     2._rp_, 1._rp_,10._rp_, 1._rp_, 3._rp_,                    &
                     1._rp_, 2._rp_, 1._rp_,10._rp_, 2._rp_,                    &
                     0._rp_, 0._rp_, 3._rp_, 2._rp_,10._rp_ ], [n,n] )
      b = [ 1._rp_,2._rp_,3._rp_,4._rp_,5._rp_ ]
      CALL set_node( node(1), 2_ip_, [1_ip_,2_ip_,3_ip_,4_ip_], 2_ip_ )
      CALL set_node( node(2), 3_ip_, [3_ip_,4_ip_,5_ip_], 0_ip_ )
      CALL set_a( node(1), [1_ip_,2_ip_,3_ip_,4_ip_,2_ip_,3_ip_,4_ip_],        &
                           [1_ip_,1_ip_,1_ip_,1_ip_,2_ip_,2_ip_,2_ip_],        &
                  [A(1,1),A(2,1),A(3,1),A(4,1),A(2,2),A(3,2),A(4,2)] )
      CALL set_a( node(2), [1_ip_,2_ip_,3_ip_,2_ip_,3_ip_,3_ip_],              &
                           [1_ip_,1_ip_,1_ip_,2_ip_,2_ip_,3_ip_],              &
                  [A(3,3),A(4,3),A(5,3),A(4,4),A(5,4),A(5,5)] )
      CALL solve_check( node, 2_ip_, A, b, n, n, "2-node SPD (Cholesky posdef)", &
                        nfail, posdef = .TRUE. )
   END SUBROUTINE case_posdef

   SUBROUTINE case_2node_spd( nfail )
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      INTEGER( ip_ ), PARAMETER :: n = 5
      REAL( rp_ ) :: A(n,n), b(n)
      TYPE( dmf_node ) :: node(2)
      A = RESHAPE( [ 10._rp_, 1._rp_, 2._rp_, 1._rp_, 0._rp_,                   &
                     1._rp_,10._rp_, 1._rp_, 2._rp_, 0._rp_,                    &
                     2._rp_, 1._rp_,10._rp_, 1._rp_, 3._rp_,                    &
                     1._rp_, 2._rp_, 1._rp_,10._rp_, 2._rp_,                    &
                     0._rp_, 0._rp_, 3._rp_, 2._rp_,10._rp_ ], [n,n] )
      b = [ 1._rp_,2._rp_,3._rp_,4._rp_,5._rp_ ]
      CALL set_node( node(1), 2_ip_, [1_ip_,2_ip_,3_ip_,4_ip_], 2_ip_ )
      CALL set_node( node(2), 3_ip_, [3_ip_,4_ip_,5_ip_], 0_ip_ )
      CALL set_a( node(1), [1_ip_,2_ip_,3_ip_,4_ip_,2_ip_,3_ip_,4_ip_],        &
                           [1_ip_,1_ip_,1_ip_,1_ip_,2_ip_,2_ip_,2_ip_],        &
                  [A(1,1),A(2,1),A(3,1),A(4,1),A(2,2),A(3,2),A(4,2)] )
      CALL set_a( node(2), [1_ip_,2_ip_,3_ip_,2_ip_,3_ip_,3_ip_],              &
                           [1_ip_,1_ip_,1_ip_,2_ip_,2_ip_,3_ip_],              &
                  [A(3,3),A(4,3),A(5,3),A(4,4),A(5,4),A(5,5)] )
      CALL solve_check( node, 2_ip_, A, b, n, n, "2-node SPD", nfail )
   END SUBROUTINE case_2node_spd

   SUBROUTINE case_3level_spd( nfail )
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      INTEGER( ip_ ), PARAMETER :: n = 4
      REAL( rp_ ) :: A(n,n), b(n)
      TYPE( dmf_node ) :: node(3)
      A = RESHAPE( [ 20._rp_, 2._rp_, 1._rp_, 0._rp_,                          &
                      2._rp_,20._rp_, 3._rp_, 1._rp_,                          &
                      1._rp_, 3._rp_,20._rp_, 2._rp_,                          &
                      0._rp_, 1._rp_, 2._rp_,20._rp_ ], [n,n] )
      b = [ 4._rp_,3._rp_,2._rp_,1._rp_ ]
      CALL set_node( node(1), 1_ip_, [1_ip_,2_ip_,3_ip_], 2_ip_ )
      CALL set_node( node(2), 1_ip_, [2_ip_,3_ip_,4_ip_], 3_ip_ )
      CALL set_node( node(3), 2_ip_, [3_ip_,4_ip_], 0_ip_ )
      CALL set_a( node(1), [1_ip_,2_ip_,3_ip_], [1_ip_,1_ip_,1_ip_],           &
                  [A(1,1),A(2,1),A(3,1)] )
      CALL set_a( node(2), [1_ip_,2_ip_,3_ip_], [1_ip_,1_ip_,1_ip_],           &
                  [A(2,2),A(3,2),A(4,2)] )
      CALL set_a( node(3), [1_ip_,2_ip_,2_ip_], [1_ip_,1_ip_,2_ip_],           &
                  [A(3,3),A(4,3),A(4,4)] )
      CALL solve_check( node, 3_ip_, A, b, n, n, "3-level SPD (post-assembly)", nfail )
   END SUBROUTINE case_3level_spd

   SUBROUTINE case_indef_2x2( nfail )
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      INTEGER( ip_ ), PARAMETER :: n = 2
      REAL( rp_ ) :: A(n,n), b(n)
      TYPE( dmf_node ) :: node(1)
      A = RESHAPE( [ 0.001_rp_, 1._rp_, 1._rp_, 0.001_rp_ ], [n,n] )
      b = [ 1._rp_, 2._rp_ ]
      CALL set_node( node(1), 2_ip_, [1_ip_,2_ip_], 0_ip_ )
      CALL set_a( node(1), [1_ip_,2_ip_,2_ip_], [1_ip_,1_ip_,2_ip_],           &
                  [A(1,1),A(2,1),A(2,2)] )
      CALL solve_check( node, 1_ip_, A, b, n, n, "indef 2x2", nfail )
   END SUBROUTINE case_indef_2x2

   SUBROUTINE case_delay( nfail )
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      INTEGER( ip_ ), PARAMETER :: n = 3
      REAL( rp_ ) :: A(n,n), b(n)
      TYPE( dmf_node ) :: node(2)
      ! var1 has a tiny pivot + large coupling -> delayed from leaf to root
      A = RESHAPE( [ 0.001_rp_, 1._rp_, 1._rp_,                                &
                     1._rp_,   10._rp_, 0._rp_,                                &
                     1._rp_,    0._rp_,10._rp_ ], [n,n] )
      b = [ 1._rp_, 2._rp_, 3._rp_ ]
      CALL set_node( node(1), 1_ip_, [1_ip_,2_ip_,3_ip_], 2_ip_ )
      CALL set_node( node(2), 2_ip_, [2_ip_,3_ip_], 0_ip_ )
      CALL set_a( node(1), [1_ip_,2_ip_,3_ip_], [1_ip_,1_ip_,1_ip_],           &
                  [A(1,1),A(2,1),A(3,1)] )
      CALL set_a( node(2), [1_ip_,2_ip_], [1_ip_,2_ip_], [A(2,2),A(3,3)] )
      CALL solve_check( node, 2_ip_, A, b, n, n, "forced delay (1 col)", nfail )
   END SUBROUTINE case_delay

   SUBROUTINE case_blocked( nfail )
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      INTEGER( ip_ ), PARAMETER :: n = 24
      REAL( rp_ ) :: A(n,n), B(n,n), rhs(n)
      TYPE( dmf_node ) :: node(1)
      INTEGER( ip_ ) :: i, j, k
      DO j = 1, n
         DO i = 1, n
            B(i,j) = SIN( 0.1_rp_*i + 0.3_rp_*j ) + 0.5_rp_*COS( 0.07_rp_*i*j )
         END DO
      END DO
      A = MATMUL( B, TRANSPOSE( B ) )
      DO i = 1, n
         A(i,i) = A(i,i) + REAL( n, rp_ )
      END DO
      DO i = 1, n
         rhs(i) = 1.0_rp_ + MOD( REAL( i, rp_ ), 3.0_rp_ )
      END DO
      ! single dense node; factor with a small panel width to exercise blocking
      CALL set_node( node(1), n, [ (i, i = 1, n) ], 0_ip_ )
      k = 0
      block
        integer(ip_) :: ii, jj
        integer(ip_), allocatable :: ai(:), aj(:)
        real(rp_), allocatable :: av(:)
        allocate( ai(n*(n+1)/2), aj(n*(n+1)/2), av(n*(n+1)/2) )
        do jj = 1, n
          do ii = jj, n
            k = k + 1; ai(k) = ii; aj(k) = jj; av(k) = A(ii,jj)
          end do
        end do
        call set_a( node(1), ai, aj, av )
      end block
      CALL solve_check( node, 1_ip_, A, rhs, n, 4_ip_, "blocked factor (nb=4)", nfail )
   END SUBROUTINE case_blocked

   SUBROUTINE case_child_contrib( nfail )
      INTEGER( ip_ ), INTENT( INOUT ) :: nfail
      INTEGER( ip_ ), PARAMETER :: n = 5
      REAL( rp_ ) :: A(n,n)
      TYPE( dmf_node ) :: mono(2), child(1), par(1)
      TYPE( subtree_contrib_t ) :: ct(1)
      LOGICAL :: ok
      REAL( rp_ ) :: dl
      A = RESHAPE( [ 10._rp_, 1._rp_, 2._rp_, 1._rp_, 0._rp_,                   &
                     1._rp_,10._rp_, 1._rp_, 2._rp_, 0._rp_,                    &
                     2._rp_, 1._rp_,10._rp_, 1._rp_, 3._rp_,                    &
                     1._rp_, 2._rp_, 1._rp_,10._rp_, 2._rp_,                    &
                     0._rp_, 0._rp_, 3._rp_, 2._rp_,10._rp_ ], [n,n] )
      ! monolithic
      CALL set_node( mono(1), 2_ip_, [1_ip_,2_ip_,3_ip_,4_ip_], 2_ip_ )
      CALL set_node( mono(2), 3_ip_, [3_ip_,4_ip_,5_ip_], 0_ip_ )
      CALL set_a( mono(1), [1_ip_,2_ip_,3_ip_,4_ip_,2_ip_,3_ip_,4_ip_],        &
                           [1_ip_,1_ip_,1_ip_,1_ip_,2_ip_,2_ip_,2_ip_],        &
                  [A(1,1),A(2,1),A(3,1),A(4,1),A(2,2),A(3,2),A(4,2)] )
      CALL set_a( mono(2), [1_ip_,2_ip_,3_ip_,2_ip_,3_ip_,3_ip_],              &
                           [1_ip_,1_ip_,1_ip_,2_ip_,2_ip_,3_ip_],              &
                  [A(3,3),A(4,3),A(5,3),A(4,4),A(5,4),A(5,5)] )
      CALL factor_subtree_delay( mono, 2_ip_, n, .TRUE., 0.01_rp_,             &
                                 EPSILON(1.0_rp_), n, .FALSE., ok )
      ! split: leaf as its own subtree, contribution fed to the root
      CALL set_node( child(1), 2_ip_, [1_ip_,2_ip_,3_ip_,4_ip_], 0_ip_ )
      CALL set_a( child(1), [1_ip_,2_ip_,3_ip_,4_ip_,2_ip_,3_ip_,4_ip_],       &
                            [1_ip_,1_ip_,1_ip_,1_ip_,2_ip_,2_ip_,2_ip_],       &
                  [A(1,1),A(2,1),A(3,1),A(4,1),A(2,2),A(3,2),A(4,2)] )
      CALL factor_subtree_delay( child, 1_ip_, n, .TRUE., 0.01_rp_,            &
                                 EPSILON(1.0_rp_), n, .FALSE., ok )
      CALL extract_contrib( child(1), ct(1) )
      CALL set_node( par(1), 3_ip_, [3_ip_,4_ip_,5_ip_], 0_ip_ )
      CALL set_a( par(1), [1_ip_,2_ip_,3_ip_,2_ip_,3_ip_,3_ip_],               &
                          [1_ip_,1_ip_,1_ip_,2_ip_,2_ip_,3_ip_],               &
                  [A(3,3),A(4,3),A(5,3),A(4,4),A(5,4),A(5,5)] )
      par(1)%contribs = [ 1_ip_ ]
      CALL factor_subtree_delay( par, 1_ip_, n, .TRUE., 0.01_rp_,              &
                                 EPSILON(1.0_rp_), n, .FALSE., ok, contribs = ct )
      dl = MAXVAL( ABS( mono(2)%lcol( 1:mono(2)%nrow, 1:mono(2)%ncol ) -       &
                        par(1)%lcol( 1:par(1)%nrow, 1:par(1)%ncol ) ) )
      IF ( par(1)%nelim == mono(2)%nelim .AND.                                 &
           dl <= SQRT( EPSILON( 1.0_rp_ ) ) ) THEN
         WRITE( 6, "( '  ok   : child_contrib == monolithic' )" )
      ELSE
         WRITE( 6, "( '  FAIL : child_contrib, dL=', ES10.3 )" ) dl
         nfail = nfail + 1
      END IF
   END SUBROUTINE case_child_contrib

   END PROGRAM GALAHAD_SSIDS_factor_test_program
