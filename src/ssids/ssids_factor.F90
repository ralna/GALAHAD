! THIS VERSION: GALAHAD 5.5 - 2026-07-27
!
! Pure-Fortran SSIDS factor/solve kernels + a serial multifrontal driver,
! ported to GALAHAD templated precision from the C++ SPRAL reference (the former
! src/ssids/cpu C++ backend). The bodies are transcribed close to the C++ so the
! numerics match; only the kind imports differ (GALAHAD ip_/rp_/long_ mapped to
! the port's local kind names).
!
! Contents:
!   * calc_ld, ldlt_tpp_factor (+ helpers)           -- dense block LDL^T (TPP)
!   * block_ldlt (+ helpers)                         -- Bunch-Kaufman full block
!   * ldlt_app_factor (+ helpers)                    -- a-posteriori pivoted (APP)
!                                                       blocked LDL^T (verbatim
!                                                       port of C++ ldlt_app)
!   * factor_node_indef                              -- node factor + contrib
!                                                       (APP if nb<n, else TPP)
!   * assemble_expected / assemble_expected_contrib  -- child -> parent assembly
!   * ldlt_app_solve_fwd/diag/bwd                    -- per-node solves
!   * dmf_node, subtree_contrib_t                    -- multifrontal state
!   * factor_subtree_delay                           -- factor incl. delayed pivots
!                                                       + foreign child_contrib
!   * subtree_solve_fwd/diag/bwd_delay               -- tree solves (multi-RHS)
!   * extract_contrib                                -- produce a child_contrib
!
! These kernels are driven by GALAHAD_SSIDS_numeric_subtree_precision, which
! wires them into the SSIDS factorization/solve path.

#include "galahad_modules.h"
#include "galahad_blas.h"
#include "galahad_lapack.h"

!-*-*-  G A L A H A D _ S S I D S _ F A C T O R   M O D U L E  -*-*-*-*-

 MODULE GALAHAD_SSIDS_factor_precision
   USE GALAHAD_KINDS_precision, ONLY : ip_, rp_, long_
   USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY : IEEE_VALUE, IEEE_POSITIVE_INF,   &
                                             IEEE_IS_FINITE
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: dmf_node, subtree_contrib_t, factor_subtree_delay, extract_contrib
   PUBLIC :: subtree_solve_fwd_delay, subtree_solve_diag_delay,               &
             subtree_solve_bwd_delay
   ! low-level kernels exposed for per-routine unit testing (ssids_factort)
   PUBLIC :: calc_ld, ldlt_tpp_factor, ldlt_blocked_factor, factor_node_indef, &
             block_ldlt, ldlt_app_factor,                                     &
             assemble_expected, assemble_expected_contrib,                    &
             ldlt_app_solve_fwd, ldlt_app_solve_diag, ldlt_app_solve_bwd

   ! factor-routine return codes (the `flag` argument); 0 = success
   INTEGER(ip_), PARAMETER :: FLAG_SINGULAR = -1  ! singular pivot / not definite
   INTEGER(ip_), PARAMETER :: FLAG_OOM      = -2  ! allocation failure

   TYPE subtree_contrib_t
      INTEGER(ip_) :: cn = 0
      INTEGER(ip_) :: ndelay = 0
      INTEGER(ip_), ALLOCATABLE :: rlist(:)
      INTEGER(ip_), ALLOCATABLE :: delay_perm(:)
      REAL(rp_),    ALLOCATABLE :: val(:,:)
      REAL(rp_),    ALLOCATABLE :: delay_val(:,:)
   END TYPE subtree_contrib_t

   TYPE dmf_node
      INTEGER(ip_) :: symb_ncol = 0, symb_nrow = 0, parent = 0
      INTEGER(ip_) :: ndelay_in = 0, ncol = 0, nrow = 0, ldl = 0
      INTEGER(ip_) :: nelim = 0, ndelay_out = 0
      INTEGER(ip_) :: nfirst = 0, nsecond = 0  ! cols not elim by 1st/2nd pass
      INTEGER(ip_), ALLOCATABLE :: rlist(:), perm(:)
      REAL(rp_),    ALLOCATABLE :: lcol(:,:), d(:), contrib(:,:)
      INTEGER(ip_), ALLOCATABLE :: ai(:), aj(:)
      REAL(rp_),    ALLOCATABLE :: av(:)
      INTEGER(ip_), ALLOCATABLE :: contribs(:)
   END TYPE dmf_node

   ! per-thread scratch reused across node tasks (avoids per-node allocation)
   INTEGER(ip_), ALLOCATABLE :: tls_pmap(:)
   !$omp threadprivate(tls_pmap)

 CONTAINS

! ============================ calc_ld ==================================
   SUBROUTINE calc_ld(op_t, m, n, l, ldl, d, ld, ldld)
      LOGICAL,     INTENT(IN)  :: op_t
      INTEGER(ip_), INTENT(IN)  :: m, n, ldl, ldld
      REAL(rp_),    INTENT(IN)  :: l(ldl, *), d(*)
      REAL(rp_),    INTENT(OUT) :: ld(ldld, *)
      INTEGER(ip_) :: col, row
      REAL(rp_)    :: d11, d21, d22, det, a1, a2
      col = 0
      DO WHILE (col < n)
         IF (col+1 == n .OR. IEEE_IS_FINITE(d(2*col+3))) THEN
            d11 = d(2*col+1)
            IF (d11 /= 0.0_rp_) d11 = 1.0_rp_/d11
            IF (.NOT. op_t) THEN
               !$omp simd
               DO row = 0, m-1
                  ld(row+1, col+1) = d11 * l(row+1, col+1)
               END DO
            ELSE
               DO row = 0, m-1
                  ld(row+1, col+1) = d11 * l(col+1, row+1)
               END DO
            END IF
            col = col + 1
         ELSE
            d11 = d(2*col+1); d21 = d(2*col+2); d22 = d(2*col+4)
            det = d11*d22 - d21*d21
            d11 = d11/det; d21 = d21/det; d22 = d22/det
            DO row = 0, m-1
               IF (.NOT. op_t) THEN
                  a1 = l(row+1, col+1); a2 = l(row+1, col+2)
               ELSE
                  a1 = l(col+1, row+1); a2 = l(col+2, row+1)
               END IF
               ld(row+1, col+1) =  d22*a1 - d21*a2
               ld(row+1, col+2) = -d21*a1 + d11*a2
            END DO
            col = col + 2
         END IF
      END DO
   END SUBROUTINE calc_ld

! ============================ ldlt_tpp ================================
   LOGICAL FUNCTION check_col_small(idx, from, to, a, lda, small)
      INTEGER(ip_), INTENT(IN) :: idx, from, to, lda
      REAL(rp_),    INTENT(IN) :: a(lda, *), small
      INTEGER(ip_) :: c, r
      check_col_small = .TRUE.
      DO c = from, idx-1
         check_col_small = check_col_small .AND. (ABS(a(idx+1, c+1)) < small)
      END DO
      DO r = idx, to-1
         check_col_small = check_col_small .AND. (ABS(a(r+1, idx+1)) < small)
      END DO
   END FUNCTION check_col_small

   INTEGER(ip_) FUNCTION find_row_abs_max(from, to, prow, a, lda) RESULT(best_idx)
      INTEGER(ip_), INTENT(IN) :: from, to, prow, lda
      REAL(rp_),    INTENT(IN) :: a(lda, *)
      INTEGER(ip_) :: idx
      REAL(rp_)    :: best_val
      IF (from >= to) THEN
         best_idx = -1; RETURN
      END IF
      best_idx = from
      best_val = ABS(a(prow+1, from+1))
      DO idx = from+1, to-1
         IF (ABS(a(prow+1, idx+1)) > best_val) THEN
            best_idx = idx; best_val = ABS(a(prow+1, idx+1))
         END IF
      END DO
   END FUNCTION find_row_abs_max

   SUBROUTINE swap_cols(col1i, col2i, m, perm, a, lda, nleft, aleft, ldleft)
      INTEGER(ip_), INTENT(IN)    :: col1i, col2i, m, lda, nleft, ldleft
      INTEGER(ip_), INTENT(INOUT) :: perm(*)
      REAL(rp_),    INTENT(INOUT) :: a(lda, *), aleft(ldleft, *)
      INTEGER(ip_) :: col1, col2, c, i, r, itmp
      REAL(rp_)    :: rtmp
      IF (col1i == col2i) RETURN
      col1 = MIN(col1i, col2i); col2 = MAX(col1i, col2i)
      itmp = perm(col1+1); perm(col1+1) = perm(col2+1); perm(col2+1) = itmp
      DO c = 0, nleft-1
         rtmp = aleft(col1+1, c+1)
         aleft(col1+1, c+1) = aleft(col2+1, c+1); aleft(col2+1, c+1) = rtmp
      END DO
      DO c = 0, col1-1
         rtmp = a(col1+1, c+1); a(col1+1, c+1) = a(col2+1, c+1); a(col2+1, c+1) = rtmp
      END DO
      DO i = col1+1, col2-1
         rtmp = a(i+1, col1+1); a(i+1, col1+1) = a(col2+1, i+1); a(col2+1, i+1) = rtmp
      END DO
      DO r = col2+1, m-1
         rtmp = a(r+1, col1+1); a(r+1, col1+1) = a(r+1, col2+1); a(r+1, col2+1) = rtmp
      END DO
      rtmp = a(col1+1, col1+1); a(col1+1, col1+1) = a(col2+1, col2+1)
      a(col2+1, col2+1) = rtmp
   END SUBROUTINE swap_cols

   REAL(rp_) FUNCTION find_rc_abs_max_exclude(col, nelim, m, a, lda, exclude) &
         RESULT(best)
      INTEGER(ip_), INTENT(IN) :: col, nelim, m, lda, exclude
      REAL(rp_),    INTENT(IN) :: a(lda, *)
      INTEGER(ip_) :: c, r
      best = 0.0_rp_
      DO c = nelim, col-1
         IF (c == exclude) CYCLE
         best = MAX(best, ABS(a(col+1, c+1)))
      END DO
      DO r = col+1, m-1
         IF (r == exclude) CYCLE
         best = MAX(best, ABS(a(r+1, col+1)))
      END DO
   END FUNCTION find_rc_abs_max_exclude

   LOGICAL FUNCTION test_2x2(t, p, maxt, maxp, a, lda, u, small, d, nelim) &
         RESULT(ok)
      INTEGER(ip_), INTENT(IN)    :: t, p, lda, nelim
      REAL(rp_),    INTENT(IN)    :: maxt, maxp, a(lda, *), u, small
      REAL(rp_),    INTENT(INOUT) :: d(*)
      REAL(rp_) :: a11, a21, a22, detpiv, detpiv0, detpiv1, detscale, maxpiv, x1, x2
      a11 = a(t+1, t+1); a21 = a(p+1, t+1); a22 = a(p+1, p+1)
      maxpiv = MAX(ABS(a11), ABS(a21), ABS(a22))
      ok = .FALSE.
      IF (maxpiv < small) RETURN
      detscale = 1.0_rp_ / maxpiv
      detpiv0 = (a11*detscale)*a22
      detpiv1 = (a21*detscale)*a21
      detpiv = detpiv0 - detpiv1
      IF (ABS(detpiv) < MAX(small, MAX(ABS(detpiv0/2), ABS(detpiv1/2)))) RETURN
      d(2*nelim+1) = (a22*detscale)/detpiv
      d(2*nelim+2) = (-a21*detscale)/detpiv
      d(2*nelim+3) = IEEE_VALUE(1.0_rp_, IEEE_POSITIVE_INF)
      d(2*nelim+4) = (a11*detscale)/detpiv
      IF (MAX(maxt, maxp) < small) THEN       ! match C++ strict-< boundary
         ok = .TRUE.; RETURN
      END IF
      x1 = ABS(d(2*nelim+1))*maxt + ABS(d(2*nelim+2))*maxp
      x2 = ABS(d(2*nelim+2))*maxt + ABS(d(2*nelim+4))*maxp
      IF (u*MAX(x1, x2) < 1.0_rp_) ok = .TRUE.  ! match C++ form u*x < 1
   END FUNCTION test_2x2

   SUBROUTINE apply_2x2(nelim, m, a, lda, ld, ldld, d)
      INTEGER(ip_), INTENT(IN)    :: nelim, m, lda, ldld
      REAL(rp_),    INTENT(INOUT) :: a(lda, *), ld(ldld, *)
      REAL(rp_),    INTENT(IN)    :: d(*)
      INTEGER(ip_) :: r
      REAL(rp_)    :: d11, d21, d22
      a(nelim+1, nelim+1) = 1.0_rp_
      a(nelim+2, nelim+1) = 0.0_rp_
      a(nelim+2, nelim+2) = 1.0_rp_
      d11 = d(2*nelim+1); d21 = d(2*nelim+2); d22 = d(2*nelim+4)
      !$omp simd
      DO r = nelim+2, m-1
         ld(r+1, 1) = a(r+1, nelim+1)
         ld(r+1, 2) = a(r+1, nelim+2)
         a(r+1, nelim+1) = d11*ld(r+1, 1) + d21*ld(r+1, 2)
         a(r+1, nelim+2) = d21*ld(r+1, 1) + d22*ld(r+1, 2)
      END DO
   END SUBROUTINE apply_2x2

   SUBROUTINE apply_1x1(nelim, m, a, lda, ld, ldld, d)
      INTEGER(ip_), INTENT(IN)    :: nelim, m, lda, ldld
      REAL(rp_),    INTENT(INOUT) :: a(lda, *), ld(ldld, *)
      REAL(rp_),    INTENT(IN)    :: d(*)
      INTEGER(ip_) :: r
      REAL(rp_)    :: d11
      a(nelim+1, nelim+1) = 1.0_rp_
      d11 = d(2*nelim+1)
      !$omp simd
      DO r = nelim+1, m-1
         ld(r+1, 1) = a(r+1, nelim+1)
         a(r+1, nelim+1) = d11*a(r+1, nelim+1)
      END DO
   END SUBROUTINE apply_1x1

   SUBROUTINE zero_col(col, m, a, lda)
      INTEGER(ip_), INTENT(IN)    :: col, m, lda
      REAL(rp_),    INTENT(INOUT) :: a(lda, *)
      INTEGER(ip_) :: r
      DO r = col, m-1
         a(r+1, col+1) = 0.0_rp_
      END DO
   END SUBROUTINE zero_col

   FUNCTION ldlt_tpp_factor(m, n, perm, a, lda, d, ld, ldld, action, u, small, &
         nleft, aleft, ldleft, flag) RESULT(nelim)
      INTEGER(ip_), INTENT(IN)    :: m, n, lda, ldld, nleft, ldleft
      INTEGER(ip_), INTENT(INOUT) :: perm(*)
      REAL(rp_),    INTENT(INOUT) :: a(lda, *), d(*), ld(ldld, *), aleft(ldleft, *)
      LOGICAL,     INTENT(IN)    :: action
      REAL(rp_),    INTENT(IN)    :: u, small
      INTEGER(ip_), INTENT(OUT)   :: flag
      INTEGER(ip_)                :: nelim
      INTEGER(ip_) :: p, t
      REAL(rp_)    :: maxt, maxp
      nelim = 0; flag = 0
      DO WHILE (nelim < n)
         IF (check_col_small(nelim, nelim, m, a, lda, small)) THEN
            IF (.NOT. action) THEN
               flag = FLAG_SINGULAR; RETURN         ! singular, action=.FALSE. -> abort
            END IF
            CALL swap_cols(nelim, nelim, m, perm, a, lda, nleft, aleft, ldleft)
            CALL zero_col(nelim, m, a, lda)
            d(2*nelim+1) = 0.0_rp_; d(2*nelim+2) = 0.0_rp_
            nelim = nelim + 1; CYCLE
         END IF
         DO p = nelim+1, n-1
            IF (check_col_small(p, nelim, m, a, lda, small)) THEN
               IF (.NOT. action) THEN
                  flag = FLAG_SINGULAR; RETURN      ! singular, action=.FALSE. -> abort
               END IF
               CALL swap_cols(p, nelim, m, perm, a, lda, nleft, aleft, ldleft)
               CALL zero_col(nelim, m, a, lda)
               d(2*nelim+1) = 0.0_rp_; d(2*nelim+2) = 0.0_rp_
               nelim = nelim + 1; EXIT
            END IF
            t = find_row_abs_max(nelim, p, p, a, lda)
            maxt = find_rc_abs_max_exclude(t, nelim, m, a, lda, p)
            maxp = find_rc_abs_max_exclude(p, nelim, m, a, lda, t)
            IF (test_2x2(t, p, maxt, maxp, a, lda, u, small, d, nelim)) THEN
               CALL swap_cols(t, nelim,   m, perm, a, lda, nleft, aleft, ldleft)
               CALL swap_cols(p, nelim+1, m, perm, a, lda, nleft, aleft, ldleft)
               CALL apply_2x2(nelim, m, a, lda, ld, ldld, d)
               CALL DGEMM('N', 'T', m-nelim-2, n-nelim-2, 2_ip_, -1.0_rp_, &
                    a(nelim+3, nelim+1), lda, ld(nelim+3, 1), ldld, 1.0_rp_, &
                    a(nelim+3, nelim+3), lda)
               nelim = nelim + 2; EXIT
            END IF
            maxp = MAX(maxp, ABS(a(p+1, t+1)))
            IF (ABS(a(p+1, p+1)) >= u*maxp) THEN
               CALL swap_cols(p, nelim, m, perm, a, lda, nleft, aleft, ldleft)
               d(2*nelim+1) = 1.0_rp_ / a(nelim+1, nelim+1)
               d(2*nelim+2) = 0.0_rp_
               CALL apply_1x1(nelim, m, a, lda, ld, ldld, d)
               CALL DGEMM('N', 'T', m-nelim-1, n-nelim-1, 1_ip_, -1.0_rp_, &
                    a(nelim+2, nelim+1), lda, ld(nelim+2, 1), ldld, 1.0_rp_, &
                    a(nelim+2, nelim+2), lda)
               nelim = nelim + 1; EXIT
            END IF
         END DO
         IF (p >= n) THEN
            p = nelim
            maxp = find_rc_abs_max_exclude(p, nelim, m, a, lda, -1_ip_)
            IF (ABS(a(p+1, p+1)) >= u*maxp) THEN
               CALL swap_cols(p, nelim, m, perm, a, lda, nleft, aleft, ldleft)
               d(2*nelim+1) = 1.0_rp_ / a(nelim+1, nelim+1)
               d(2*nelim+2) = 0.0_rp_
               CALL apply_1x1(nelim, m, a, lda, ld, ldld, d)
               CALL DGEMM('N', 'T', m-nelim-1, n-nelim-1, 1_ip_, -1.0_rp_, &
                    a(nelim+2, nelim+1), lda, ld(nelim+2, 1), ldld, 1.0_rp_, &
                    a(nelim+2, nelim+2), lda)
               nelim = nelim + 1
            ELSE
               EXIT
            END IF
         END IF
      END DO
   END FUNCTION ldlt_tpp_factor

! ========================= ldlt_blocked ==============================
   !> Blocked RIGHT-looking LDL^T-TPP with intra-front OpenMP parallelism -- the
   !! practical core of the C++ ldlt_app. Each block column (width nb) is factored
   !! by ldlt_tpp (identical threshold pivoting), then the INDEPENDENT trailing
   !! block-column updates are run as !$omp tasks, so a single large front is
   !! factored in parallel. The tasks bind to the enclosing team (the tree DAG in
   !! factor_subtree_delay) so they compose without oversubscription; with no
   !! enclosing parallel region they run serially. A delayed pivot finishes the
   !! tail with a full ldlt_tpp (delays match the unblocked kernel). nb >= n
   !! reduces to a single unblocked call.
   FUNCTION ldlt_blocked_factor(m, n, perm, a, lda, d, u, small, action, nb, &
                                flag) RESULT(nelim)
      INTEGER(ip_), INTENT(IN)    :: m, n, lda, nb
      INTEGER(ip_), INTENT(INOUT) :: perm(*)
      REAL(rp_),    INTENT(INOUT) :: a(lda, *), d(*)
      REAL(rp_),    INTENT(IN)    :: u, small
      LOGICAL,     INTENT(IN)    :: action
      INTEGER(ip_), INTENT(OUT)   :: flag
      INTEGER(ip_)                :: nelim
      INTEGER(ip_) :: ps, pe, kf, nrem, js, je
      REAL(rp_), ALLOCATABLE :: ldw(:, :)
      nelim = 0; flag = 0; ps = 0
      DO WHILE (ps < n)
         pe = MIN(ps + nb, n)
         ! factor the tall panel [ps:m, ps:pe] (up to date -- right-looking)
         ALLOCATE(ldw(m-ps, 2))
         kf = ldlt_tpp_factor(m-ps, pe-ps, perm(ps+1), a(ps+1, ps+1), lda, &
                              d(2*ps+1), ldw, m-ps, action, u, small, ps, &
                              a(ps+1, 1), lda, flag)
         DEALLOCATE(ldw)
         nelim = nelim + kf
         IF (flag /= 0) RETURN
         IF (kf < pe - ps) THEN
            ! delayed pivot: bring the untouched trailing up to date, finish tail
            IF (pe < n) THEN
               nrem = n - pe
               CALL update_trailing_block(m, ps, pe, n, a, lda, d, nelim)
               ALLOCATE(ldw(m-nelim, 2))
               kf = ldlt_tpp_factor(m-nelim, n-nelim, perm(nelim+1), &
                                    a(nelim+1, nelim+1), lda, d(2*nelim+1), ldw, &
                                    m-nelim, action, u, small, nelim, &
                                    a(nelim+1, 1), lda, flag)
               DEALLOCATE(ldw)
               nelim = nelim + kf
            END IF
            RETURN
         END IF
         ! right-looking: update each trailing block column [pe,n) independently
         DO js = pe, n-1, nb
            je = MIN(js + nb, n)
            !$omp task default(shared) firstprivate(js, je) if(n-pe > nb)
            CALL update_trailing_block(m, ps, js, je, a, lda, d, nelim)
            !$omp end task
         END DO
         !$omp taskwait
         ps = pe
      END DO
   END FUNCTION ldlt_blocked_factor

   !> Trailing update of block column [cs,ce) by the pivots [ps0,nelim):
   !! a[cs:m, cs:ce] -= L[cs:m, ps0:nelim] * D * L[cs:ce, ps0:nelim]^T.
   SUBROUTINE update_trailing_block(m, ps0, cs, ce, a, lda, d, nelim)
      INTEGER(ip_), INTENT(IN)    :: m, ps0, cs, ce, lda, nelim
      REAL(rp_),    INTENT(INOUT) :: a(lda, *)
      REAL(rp_),    INTENT(IN)    :: d(*)
      INTEGER(ip_) :: k, cw
      REAL(rp_), ALLOCATABLE :: ld2(:, :)
      k = nelim - ps0
      IF (k <= 0) RETURN
      cw = ce - cs
      ALLOCATE(ld2(cw, k))
      CALL calc_ld(.FALSE., cw, k, a(cs+1, ps0+1), lda, d(2*ps0+1), ld2, cw)
      CALL DGEMM('N', 'T', m-cs, cw, k, -1.0_rp_, a(cs+1, ps0+1), lda, ld2, cw, &
                 1.0_rp_, a(cs+1, cs+1), lda)
      DEALLOCATE(ld2)
   END SUBROUTINE update_trailing_block

! ===================== block_ldlt + a-posteriori pivoted (APP) =========
!====================== block_ldlt (validated) ==========================
  SUBROUTINE bk_find_maxloc(from, a, lda, bs, bestv, rloc, cloc)
    INTEGER(ip_), INTENT(IN)  :: from, lda, bs
    REAL(rp_),    INTENT(IN)  :: a(lda,*)
    REAL(rp_),    INTENT(OUT) :: bestv
    INTEGER(ip_), INTENT(OUT) :: rloc, cloc
    INTEGER(ip_) :: c, r
    REAL(rp_)    :: bv, cmax
    ! per-column max reduction (vectorisable) + scalar locate of the first row
    ! achieving it; identical (c,r) tie-break to the scalar double loop.
    bv = -1.0_rp_; rloc = bs; cloc = bs
    DO c = from, bs-1
      cmax = -1.0_rp_
      !$omp simd reduction(max: cmax)
      DO r = c, bs-1
        cmax = MAX(cmax, ABS(a(r+1, c+1)))
      END DO
      IF (cmax > bv) THEN
        DO r = c, bs-1
          IF (ABS(a(r+1, c+1)) == cmax) THEN
            bv = cmax; rloc = r; cloc = c; EXIT
          END IF
        END DO
      END IF
    END DO
    IF (cloc < bs .AND. rloc < bs) THEN
      bestv = a(rloc+1, cloc+1)
    ELSE
      bestv = 0.0_rp_
    END IF
  END SUBROUTINE bk_find_maxloc

  LOGICAL FUNCTION bk_test_2x2(a11, a21, a22, detpiv, detscale) RESULT(ok)
    REAL(rp_), INTENT(IN)  :: a11, a21, a22
    REAL(rp_), INTENT(OUT) :: detpiv, detscale
    detscale = 1.0_rp_/ABS(a21)
    detpiv = (a11*detscale)*a22 - ABS(a21)
    ok = (ABS(detpiv) >= ABS(a21)/2.0_rp_)
  END FUNCTION bk_test_2x2

  SUBROUTINE bk_swap_cols(idx1i, idx2i, n, a, lda, ldwork, bs, perm)
    INTEGER(ip_), INTENT(IN)    :: idx1i, idx2i, n, lda, bs
    REAL(rp_),    INTENT(INOUT) :: a(lda,*), ldwork(bs,*)
    INTEGER(ip_), INTENT(INOUT) :: perm(*)
    INTEGER(ip_) :: idx1, idx2, c, i, r, it
    REAL(rp_)    :: t
    IF (idx1i == idx2i) RETURN
    idx1 = MIN(idx1i, idx2i); idx2 = MAX(idx1i, idx2i)
    it = perm(idx1+1); perm(idx1+1) = perm(idx2+1); perm(idx2+1) = it
    DO c = 0, idx1-1
      t = ldwork(idx1+1, c+1); ldwork(idx1+1, c+1) = ldwork(idx2+1, c+1)
      ldwork(idx2+1, c+1) = t
    END DO
    DO c = 0, idx1-1
      t = a(idx1+1, c+1); a(idx1+1, c+1) = a(idx2+1, c+1); a(idx2+1, c+1) = t
    END DO
    DO i = idx1+1, idx2-1
      t = a(i+1, idx1+1); a(i+1, idx1+1) = a(idx2+1, i+1); a(idx2+1, i+1) = t
    END DO
    t = a(idx1+1, idx1+1); a(idx1+1, idx1+1) = a(idx2+1, idx2+1)
    a(idx2+1, idx2+1) = t
    DO r = idx2+1, n-1
      t = a(r+1, idx1+1); a(r+1, idx1+1) = a(r+1, idx2+1); a(r+1, idx2+1) = t
    END DO
  END SUBROUTINE bk_swap_cols

  SUBROUTINE bk_update_1x1(p, a, lda, ldw, bs)
    INTEGER(ip_), INTENT(IN)    :: p, lda, bs
    REAL(rp_),    INTENT(INOUT) :: a(lda,*)
    REAL(rp_),    INTENT(IN)    :: ldw(bs,*)
    INTEGER(ip_) :: c, r
    DO c = p+1, bs-1
      !$omp simd
      DO r = c, bs-1
        a(r+1, c+1) = a(r+1, c+1) - ldw(c+1, p+1)*a(r+1, p+1)
      END DO
    END DO
  END SUBROUTINE bk_update_1x1

  SUBROUTINE bk_update_2x2(p, a, lda, ldw, bs)
    INTEGER(ip_), INTENT(IN)    :: p, lda, bs
    REAL(rp_),    INTENT(INOUT) :: a(lda,*)
    REAL(rp_),    INTENT(IN)    :: ldw(bs,*)
    INTEGER(ip_) :: c, r
    DO c = p+2, bs-1
      !$omp simd
      DO r = c, bs-1
        a(r+1, c+1) = a(r+1, c+1) - ldw(c+1, p+1)*a(r+1, p+1) &
                                  - ldw(c+1, p+2)*a(r+1, p+2)
      END DO
    END DO
  END SUBROUTINE bk_update_2x2

  SUBROUTINE block_ldlt(from, perm, a, lda, d, ldwork, action, u, small, &
                        lperm, bs, flag)
    INTEGER(ip_), INTENT(IN)    :: from, lda, bs
    INTEGER(ip_), INTENT(INOUT) :: perm(*), lperm(*)
    REAL(rp_),    INTENT(INOUT) :: a(lda,*), d(*), ldwork(bs,*)
    LOGICAL,     INTENT(IN)    :: action
    REAL(rp_),    INTENT(IN)    :: u, small
    INTEGER(ip_), INTENT(OUT)   :: flag
    INTEGER(ip_) :: p, t, m, r, it, pivsiz
    REAL(rp_)    :: bestv, a11, a21, a22, detscale, detpiv, d11, d21, d22
    flag = 0
    p = from
    DO WHILE (p < bs)
      CALL bk_find_maxloc(p, a, lda, bs, bestv, t, m)
      IF (ABS(bestv) < small) THEN
        IF (.NOT. action) THEN
          flag = FLAG_SINGULAR; RETURN            ! singular pivot, action=.FALSE. -> abort
        END IF
        DO WHILE (p < bs)
          d(2*p+1) = 0.0_rp_; d(2*p+2) = 0.0_rp_
          DO r = p, bs-1
            a(r+1, p+1) = 0.0_rp_
          END DO
          DO r = p, bs-1
            ldwork(r+1, p+1) = 0.0_rp_
          END DO
          p = p + 1
        END DO
        EXIT
      END IF
      pivsiz = 0
      IF (t == m) THEN
        a11 = a(t+1, t+1); pivsiz = 1
      ELSE
        a11 = a(m+1, m+1); a22 = a(t+1, t+1); a21 = a(t+1, m+1)
        IF (bk_test_2x2(a11, a21, a22, detpiv, detscale)) THEN
          pivsiz = 2
        ELSE
          IF (ABS(a11) > ABS(a22)) THEN
            pivsiz = 1; t = m
            IF (ABS(a11/a21) < u) pivsiz = 0
          ELSE
            pivsiz = 1; a11 = a22; m = t
            IF (ABS(a22/a21) < u) pivsiz = 0
          END IF
        END IF
      END IF
      IF (pivsiz == 0) THEN
        flag = FLAG_SINGULAR; RETURN
      ELSE IF (pivsiz == 1) THEN
        d11 = 1.0_rp_/a11
        CALL bk_swap_cols(p, t, bs, a, lda, ldwork, bs, perm)
        it = lperm(p+1); lperm(p+1) = lperm(t+1); lperm(t+1) = it
        DO r = p+1, bs-1
          ldwork(r+1, p+1) = a(r+1, p+1)
          a(r+1, p+1) = a(r+1, p+1)*d11
        END DO
        CALL bk_update_1x1(p, a, lda, ldwork, bs)
        d(2*p+1) = d11; d(2*p+2) = 0.0_rp_
        a(p+1, p+1) = 1.0_rp_
      ELSE
        CALL bk_swap_cols(p,   m, bs, a, lda, ldwork, bs, perm)
        it = lperm(p+1); lperm(p+1) = lperm(m+1); lperm(m+1) = it
        CALL bk_swap_cols(p+1, t, bs, a, lda, ldwork, bs, perm)
        it = lperm(p+2); lperm(p+2) = lperm(t+1); lperm(t+1) = it
        d11 = (a22*detscale)/detpiv
        d22 = (a11*detscale)/detpiv
        d21 = (-a21*detscale)/detpiv
        DO r = p+2, bs-1
          ldwork(r+1, p+1) = a(r+1, p+1)
          ldwork(r+1, p+2) = a(r+1, p+2)
          a(r+1, p+1) = d11*ldwork(r+1, p+1) + d21*ldwork(r+1, p+2)
          a(r+1, p+2) = d21*ldwork(r+1, p+1) + d22*ldwork(r+1, p+2)
        END DO
        CALL bk_update_2x2(p, a, lda, ldwork, bs)
        d(2*p+1) = d11; d(2*p+2) = d21
        d(2*p+3) = IEEE_VALUE(1.0_rp_, IEEE_POSITIVE_INF)
        d(2*p+4) = d22
        a(p+1, p+1) = 1.0_rp_; a(p+2, p+1) = 0.0_rp_; a(p+2, p+2) = 1.0_rp_
      END IF
      p = p + pivsiz
    END DO
  END SUBROUTINE block_ldlt

!====================== apply_pivot / check_threshold ===================
  ! Faithful port of apply_pivot<OP_N/OP_T> (ldlt_app.cxx). diag is the
  ! (permuted, factored) diagonal block base; d its packed inverse pivots.
  SUBROUTINE app_apply_pivot(op_t, m, n, ifrom, diag, ldd, d, small, a, lda)
    LOGICAL,     INTENT(IN)    :: op_t
    INTEGER(ip_), INTENT(IN)    :: m, n, ifrom, ldd, lda
    REAL(rp_),    INTENT(IN)    :: diag(ldd,*), d(*), small
    REAL(rp_),    INTENT(INOUT) :: a(lda,*)
    INTEGER(ip_) :: i, j
    REAL(rp_)    :: d11, d21, d22, a1, a2, v
    IF (.NOT. op_t) THEN
      IF (ifrom > m .OR. m <= 0 .OR. n <= 0) RETURN
      CALL DTRSM('R', 'L', 'T', 'U', m, n, 1.0_rp_, diag, ldd, a, lda)
      i = 0
      DO WHILE (i < n)
        IF (i+1 == n .OR. IEEE_IS_FINITE(d(2*i+3))) THEN
          d11 = d(2*i+1)
          IF (d11 == 0.0_rp_) THEN
            DO j = 1, m
              v = a(j, i+1)
              IF (ABS(v) < small) THEN
                a(j, i+1) = 0.0_rp_
              ELSE
                a(j, i+1) = IEEE_VALUE(1.0_rp_, IEEE_POSITIVE_INF)*v
              END IF
            END DO
          ELSE
            DO j = 1, m
              a(j, i+1) = a(j, i+1)*d11
            END DO
          END IF
          i = i + 1
        ELSE
          d11 = d(2*i+1); d21 = d(2*i+2); d22 = d(2*i+4)
          DO j = 1, m
            a1 = a(j, i+1); a2 = a(j, i+2)
            a(j, i+1) = d11*a1 + d21*a2
            a(j, i+2) = d21*a1 + d22*a2
          END DO
          i = i + 2
        END IF
      END DO
    ELSE
      IF (ifrom > n .OR. m <= 0 .OR. n-ifrom <= 0) RETURN
      CALL DTRSM('L', 'L', 'N', 'U', m, n-ifrom, 1.0_rp_, diag, ldd, &
                 a(1, ifrom+1), lda)
      i = 0
      DO WHILE (i < m)
        IF (i+1 == m .OR. IEEE_IS_FINITE(d(2*i+3))) THEN
          d11 = d(2*i+1)
          IF (d11 == 0.0_rp_) THEN
            DO j = ifrom, n-1
              v = a(i+1, j+1)
              IF (ABS(v) < small) THEN
                a(i+1, j+1) = 0.0_rp_
              ELSE
                a(i+1, j+1) = IEEE_VALUE(1.0_rp_, IEEE_POSITIVE_INF)*v
              END IF
            END DO
          ELSE
            DO j = ifrom, n-1
              a(i+1, j+1) = a(i+1, j+1)*d11
            END DO
          END IF
          i = i + 1
        ELSE
          d11 = d(2*i+1); d21 = d(2*i+2); d22 = d(2*i+4)
          DO j = ifrom, n-1
            a1 = a(i+1, j+1); a2 = a(i+2, j+1)
            a(i+1, j+1) = d11*a1 + d21*a2
            a(i+2, j+1) = d21*a1 + d22*a2
          END DO
          i = i + 2
        END IF
      END DO
    END IF
  END SUBROUTINE app_apply_pivot

  INTEGER(ip_) FUNCTION app_check_threshold(op_t, rfrom, rto, cfrom, cto, u, &
        a, lda) RESULT(least_fail)
    LOGICAL,     INTENT(IN) :: op_t
    INTEGER(ip_), INTENT(IN) :: rfrom, rto, cfrom, cto, lda
    REAL(rp_),    INTENT(IN) :: u, a(lda,*)
    INTEGER(ip_) :: i, j
    LOGICAL :: brk
    IF (.NOT. op_t) THEN
      least_fail = cto
    ELSE
      least_fail = rto
    END IF
    DO j = cfrom, cto-1
      brk = .FALSE.
      DO i = rfrom, rto-1
        IF (ABS(a(i+1, j+1)) > 1.0_rp_/u) THEN
          IF (.NOT. op_t) THEN
            least_fail = j; RETURN
          ELSE
            least_fail = MIN(least_fail, i); brk = .TRUE.; EXIT
          END IF
        END IF
      END DO
      IF (brk) CYCLE
    END DO
  END FUNCTION app_check_threshold

!====================== the serial APP driver ===========================
  ! Drop-in replacement for ldlt_blocked_factor: factor the m x n panel a(lda,*)
  ! (n fully-summed cols, rows n+1..m are contribution rows) with a-posteriori
  ! pivoting at block size bs (= INNER_BLOCK_SIZE). On exit a holds the packed
  ! factorization: eliminated cols [1,nelim] contiguous with unit-L; failed
  ! (delayed) cols [nelim+1,n] hold the Schur complement; d packed inverse
  ! pivots; perm the (block) permutation.
  RECURSIVE FUNCTION ldlt_app_factor(m, n, perm, a, lda, d, u, small, action,  &
        bs, flag, use_tasks, aggressive) RESULT(num_elim)
    INTEGER(ip_), INTENT(IN)    :: m, n, lda, bs
    INTEGER(ip_), INTENT(INOUT) :: perm(*)
    REAL(rp_),    INTENT(INOUT) :: a(lda,*), d(*)
    REAL(rp_),    INTENT(IN)    :: u, small
    LOGICAL,     INTENT(IN)    :: action
    INTEGER(ip_), INTENT(OUT)   :: flag
    LOGICAL,     INTENT(IN), OPTIONAL :: use_tasks, aggressive
    INTEGER(ip_)                :: num_elim
    INTEGER(ip_), PARAMETER :: INNER = 32   ! inner block size (block_ldlt granularity)
    LOGICAL :: lut, aggr
    INTEGER(ip_) :: nblk, mblk, blk, iblk, jblk, next_elim, nc, nr, i, j, k
    INTEGER(ip_) :: nfail, ldc, adr, adc, from_blk, nu, uflag, ast
    INTEGER(ip_), ALLOCATABLE :: cnelim(:), cdoff(:), cnpass(:), lperm(:,:), fperm(:)
    INTEGER(ip_), ALLOCATABLE :: perm_copy(:), up2d(:,:)
    LOGICAL,     ALLOCATABLE :: cfirst(:)
    REAL(rp_),    ALLOCATABLE :: bcopy(:,:), fdiag(:,:), frect(:,:)
    REAL(rp_)    :: adum(1,1)
    INTEGER(ip_) :: insert, finsert, ii, jj, jf, jins, iins, ifl, tmpi
    LOGICAL     :: aborted
    ! task-private temporaries (each OpenMP task gets its own copy)
    INTEGER(ip_) :: tnc, tnr, tdoff, tnelim, tlflag, tk, tbp, tnp
    REAL(rp_)    :: td11, td21
    LOGICAL     :: tfin
    REAL(rp_), ALLOCATABLE :: tldw(:,:)

    flag = 0
    lut = .TRUE.; IF (PRESENT(use_tasks)) lut = use_tasks
    aggr = .FALSE.; IF (PRESENT(aggressive)) aggr = aggressive
    nblk = (n-1)/bs + 1
    mblk = (m-1)/bs + 1
    ALLOCATE(cnelim(0:nblk-1), cdoff(0:nblk-1), cnpass(0:nblk-1), cfirst(0:nblk-1))
    ALLOCATE(lperm(bs, 0:nblk-1))
    ALLOCATE(bcopy(m, n), stat=ast)          ! O(m*n) -- guard against OOM
    IF (ast /= 0) THEN
       flag = FLAG_OOM; num_elim = -1
       DEALLOCATE(cnelim, cdoff, cnpass, cfirst, lperm)
       RETURN
    END IF
    bcopy = 0.0_rp_
    cnelim = 0; cdoff = 0; cnpass = 0; cfirst = .FALSE.
    next_elim = 0
    from_blk = 0
    aborted = .FALSE.

    ! ---- aggressive (APP_AGGRESSIVE): optimistic unpivoted attempt first ----
    ! Port of the C++ driver's app_aggressive branch: try an unpivoted pass; on
    ! success we are done; on a pivoting failure, roll back and resume the
    ! careful pivoted pass from the first not-fully-accepted block column.
    IF (aggr) THEN
      bcopy(1:m, 1:n) = a(1:m, 1:n)          ! full backup for restore
      ALLOCATE(perm_copy(n)); perm_copy(1:n) = perm(1:n)
      ALLOCATE(up2d(0:mblk-1, 0:nblk-1)); up2d = -1
      CALL run_unpivoted(m, n, perm, a, lda, d, u, small, action, bs, INNER,    &
             aggr, mblk, nblk, cnelim, cnpass, cfirst, cdoff, lperm, up2d, lut, &
             nu, uflag)
      IF (uflag /= 0 .OR. nu < 0) THEN
        flag = MERGE(FLAG_OOM, FLAG_SINGULAR, uflag == FLAG_OOM); num_elim = -1
        DEALLOCATE(cnelim, cdoff, cnpass, cfirst, lperm, bcopy, perm_copy, up2d)
        RETURN
      END IF
      IF (nu >= n) THEN
        ! optimistic pass fully succeeded -- no failed columns, no compaction
        num_elim = nu
        DEALLOCATE(cnelim, cdoff, cnpass, cfirst, lperm, bcopy, perm_copy, up2d)
        RETURN
      END IF
      ! partial failure: roll back and resume pivoted from block nelim_blk
      from_blk = nu / bs
      CALL restore_unpiv(from_blk, m, n, perm, a, lda, d, bs, mblk, nblk,       &
             cnelim, cdoff, lperm, bcopy, up2d, perm_copy)
      next_elim = from_blk * bs
      DEALLOCATE(perm_copy, up2d)
    END IF

    ! Parallel a-posteriori pivoted elimination (port of run_elim_pivoted); the
    ! OpenMP task DAG lives in run_pivoted. Aggressive resumes at from_blk.
    CALL run_pivoted(m, n, perm, a, lda, d, u, small, action, bs, INNER,       &
           from_blk, mblk, nblk, cnelim, cnpass, cfirst, cdoff, lperm, bcopy,  &
           next_elim, lut, flag)
    IF (flag /= 0) THEN
      num_elim = -1
      DEALLOCATE(cnelim, cdoff, cnpass, cfirst, lperm, bcopy)
      RETURN
    END IF
    num_elim = next_elim

    ! ================= post-elimination compaction ==================
    IF (num_elim < n) THEN
      nfail = n - num_elim
      ! ---- move_back: build failed perm, compact eliminated perm ----
      ALLOCATE(fperm(nfail+1)); fperm = 0    ! +1: safe base addr for empty tail
      insert = 0; finsert = 0
      DO jblk = 0, nblk-1
        nc = MIN(bs, n - jblk*bs)
        CALL move_back(nc, cnelim(jblk), perm(jblk*bs+1), perm(insert+1), &
                       fperm(finsert+1))
        insert = insert + cnelim(jblk)
        finsert = finsert + (nc - cnelim(jblk))
      END DO
      DO i = 1, nfail
        perm(num_elim+i) = fperm(i)
      END DO
      ! ---- copy_failed: extract failed diag+rect into temp buffers ----
      ldc = nfail
      ALLOCATE(fdiag(nfail, n), frect(MAX(m-n,1), MAX(nfail,1)), stat=ast)
      IF (ast /= 0) THEN                       ! OOM in the compaction buffers
         flag = FLAG_OOM; num_elim = -1
         DEALLOCATE(cnelim, cdoff, cnpass, cfirst, lperm, bcopy)
         IF (ALLOCATED(fperm)) DEALLOCATE(fperm)
         RETURN
      END IF
      fdiag = 0.0_rp_; frect = 0.0_rp_
      jf = 0; jins = 0
      DO jblk = 0, nblk-1
        ifl = jf; iins = jins
        DO iblk = jblk, nblk-1
          CALL copy_failed_diag(iblk, jblk, blk_ncol(iblk,n,bs), &
                 blk_ncol(jblk,n,bs), cnelim(iblk), cnelim(jblk), &
                 a, lda, fdiag, ldc, num_elim, nfail, jins, ifl, iins, jf, bs)
          iins = iins + cnelim(iblk)
          ifl = ifl + (blk_ncol(iblk,n,bs) - cnelim(iblk))
        END DO
        ! rectangular part (rows >= n)
        IF (m > n) THEN
          CALL copy_failed_rect(jblk, cnelim(jblk), blk_ncol(jblk,n,bs), &
                 a, lda, frect, m-n, jf, n, m, bs)
        END IF
        jf = jf + (blk_ncol(jblk,n,bs) - cnelim(jblk))
        jins = jins + cnelim(jblk)
      END DO
      ! ---- move_up: compact eliminated columns' data ----
      jins = 0
      DO jblk = 0, nblk-1
        iins = jins
        DO iblk = jblk, nblk-1
          CALL move_up_diag(iblk, jblk, cnelim(iblk), cnelim(jblk), &
                            a, lda, iins, jins, bs)
          iins = iins + cnelim(iblk)
        END DO
        ! rect rows [n,m)
        CALL move_up_rect_all(jblk, cnelim(jblk), a, lda, jins, n, m, bs)
        jins = jins + cnelim(jblk)
      END DO
      ! ---- store failed entries back ----
      DO j = 0, n-1
        DO i = MAX(j, num_elim), n-1
          a(i+1, j+1) = fdiag(i-num_elim+1, j+1)
        END DO
      END DO
      DO j = 0, nfail-1
        DO i = 0, m-n-1
          a(n+i+1, num_elim+j+1) = frect(i+1, j+1)
        END DO
      END DO
      DEALLOCATE(fperm, fdiag, frect)
    END IF

    DEALLOCATE(cnelim, cdoff, cnpass, cfirst, lperm, bcopy)
  END FUNCTION ldlt_app_factor

  !> Parallel a-posteriori pivoted elimination (port of run_elim_pivoted): each
  !! step is an OpenMP task; block-element depend() clauses form the DAG, so the
  !! result is BIT-IDENTICAL to serial (with no OpenMP the directives collapse to
  !! serial). APP_AGGRESSIVE resumes at from_blk after a rolled-back optimistic
  !! unpivoted pass; otherwise from_blk = 0. flag returns 0 / FLAG_SINGULAR /
  !! FLAG_OOM; next_elim accumulates the eliminated-pivot count.
  RECURSIVE SUBROUTINE run_pivoted(m, n, perm, a, lda, d, u, small, action, bs, &
        inner, from_blk, mblk, nblk, cnelim, cnpass, cfirst, cdoff, lperm,      &
        bcopy, next_elim, lut, flag)
    INTEGER(ip_), INTENT(IN)    :: m, n, lda, bs, inner, from_blk, mblk, nblk
    INTEGER(ip_), INTENT(INOUT) :: perm(*), next_elim
    REAL(rp_),    INTENT(INOUT) :: a(lda,*), d(*), bcopy(m,*)
    REAL(rp_),    INTENT(IN)    :: u, small
    LOGICAL,      INTENT(IN)    :: action, lut
    INTEGER(ip_), INTENT(INOUT) :: cnelim(0:*), cnpass(0:*), cdoff(0:*)
    INTEGER(ip_), INTENT(INOUT) :: lperm(bs, 0:*)
    LOGICAL,      INTENT(INOUT) :: cfirst(0:*)
    INTEGER(ip_), INTENT(OUT)   :: flag
    INTEGER(ip_) :: blk, iblk, jblk, adr, adc
    INTEGER(ip_) :: tnc, tnr, tdoff, tnelim, tlflag, tk, tbp, tnp
    REAL(rp_)    :: td11, td21, adum(1,1)
    LOGICAL      :: tfin, aborted
    REAL(rp_), ALLOCATABLE :: tldw(:,:)
    flag = 0; aborted = .FALSE.
    !$omp taskgroup
    DO blk = from_blk, nblk-1
      ! ---- factor diagonal ----
      !$omp task if(lut) default(shared) firstprivate(blk)                             &
      !$omp   private(tnc, tnr, tdoff, tnelim, tldw, tlflag, tk)               &
      !$omp   depend(inout: a(blk*bs+1, blk*bs+1)) depend(inout: perm(blk*bs+1))
      IF (.NOT. aborted) THEN
        tnc = MIN(bs, n - blk*bs); tnr = MIN(bs, m - blk*bs)
        tdoff = 2*next_elim; cdoff(blk) = tdoff
        CALL bkp_create(blk, blk, a, lda, bcopy, m, bs, m, n)
        DO tk = 1, tnc
          lperm(tk, blk) = tk-1
        END DO
        IF (bs > INNER) THEN
          ! recurse: factor the diagonal block with the inner APP (block_ldlt on
          ! INNER-blocks + BLAS-3), inner runs serially. Outer keeps wide blocks
          ! so the apply/update GEMMs are bs-wide (matches C++ block_size).
          tnelim = ldlt_app_factor(tnr, tnc, lperm(1,blk), a(blk*bs+1,blk*bs+1), &
                     lda, d(tdoff+1), u, small, action, INNER, tlflag, .FALSE.)
          IF (tlflag /= 0) THEN
            !$omp atomic write
            aborted = .TRUE.
            flag = MERGE(FLAG_OOM, FLAG_SINGULAR, tlflag == FLAG_OOM)   ! keep OOM(2) vs abort(1)
          ELSE
            CALL permute_blkperm(perm, blk, bs, tnc, lperm(1,blk))
          END IF
        ELSE IF (tnc < bs) THEN
          ALLOCATE(tldw(tnr+2, 2))       ! +2 rows: benign tail-DGEMM base addr
          tnelim = ldlt_tpp_factor(tnr, tnc, lperm(1,blk), a(blk*bs+1,blk*bs+1),&
                     lda, d(tdoff+1), tldw, tnr+2, action, u, small, 0_ip_, adum,&
                     1_ip_, tlflag)
          DEALLOCATE(tldw)
          IF (tlflag /= 0) THEN        ! singular (flag<0) with action=.FALSE.
            !$omp atomic write
            aborted = .TRUE.
            flag = FLAG_SINGULAR
          ELSE
            CALL permute_blkperm(perm, blk, bs, tnc, lperm(1,blk))
          END IF
        ELSE
          ALLOCATE(tldw(bs, bs))
          CALL block_ldlt(0_ip_, perm(blk*bs+1), a(blk*bs+1,blk*bs+1), lda,      &
                          d(tdoff+1), tldw, action, u, small, lperm(1,blk), bs, &
                          tlflag)
          DEALLOCATE(tldw); tnelim = bs
          IF (tlflag /= 0) THEN        ! singular (flag<0) with action=.FALSE.
            !$omp atomic write
            aborted = .TRUE.
            flag = FLAG_SINGULAR
          END IF
        END IF
        cnelim(blk) = tnelim             ! raw block nelim (apply phase)
        cnpass(blk) = tnelim             ! init_passed(nelim)
      END IF
      !$omp end task

      ! ---- apply pivot to eliminated ROW (left blocks jblk<blk) ----
      DO jblk = 0, blk-1
        !$omp task if(lut) default(shared) firstprivate(blk, jblk) private(tbp, tnc)    &
        !$omp   depend(in: a(blk*bs+1, blk*bs+1))                               &
        !$omp   depend(inout: a(blk*bs+1, jblk*bs+1)) depend(in: perm(blk*bs+1))
        IF (.NOT. aborted) THEN
          tnc = MIN(bs, n - blk*bs)
          CALL bkp_rperm(blk, jblk, a, lda, bcopy, m, bs, n, lperm(1,blk), tnc)
          tbp = apply_T(blk, jblk, a, lda, d, cdoff, cnelim, m, n, bs, small, u)
          !$omp atomic update
          cnpass(blk) = MIN(cnpass(blk), tbp)
        END IF
        !$omp end task
      END DO
      ! ---- apply pivot to eliminated COL (below blocks iblk>blk) ----
      DO iblk = blk+1, mblk-1
        !$omp task if(lut) default(shared) firstprivate(blk, iblk) private(tbp)         &
        !$omp   depend(in: a(blk*bs+1, blk*bs+1))                               &
        !$omp   depend(inout: a(iblk*bs+1, blk*bs+1)) depend(in: perm(blk*bs+1))
        IF (.NOT. aborted) THEN
          CALL bkp_cperm(iblk, blk, a, lda, bcopy, m, bs, n, lperm(1,blk))
          tbp = apply_N(iblk, blk, a, lda, d, cdoff, cnelim, m, n, bs, small, u)
          !$omp atomic update
          cnpass(blk) = MIN(cnpass(blk), tbp)
        END IF
        !$omp end task
      END DO

      ! ---- adjust: avoid split 2x2, finalise nelim/next_elim ----
      !$omp task if(lut) default(shared) firstprivate(blk) private(tnp, td11, td21, tfin) &
      !$omp   depend(inout: perm(blk*bs+1))
      IF (.NOT. aborted) THEN
        tnp = cnpass(blk)
        IF (tnp > 0) THEN
          td11 = d(cdoff(blk) + 2*(tnp-1) + 1)
          td21 = d(cdoff(blk) + 2*(tnp-1) + 2)
          tfin = IEEE_IS_FINITE(td11)
          IF (tfin .AND. td21 /= 0.0_rp_) tnp = tnp - 1
        END IF
        cfirst(blk) = (next_elim == 0 .AND. tnp > 0)
        next_elim = next_elim + tnp
        cnelim(blk) = tnp
      END IF
      !$omp end task

      ! ---- update trailing (left of elim col) ----
      DO jblk = 0, blk-1
        DO iblk = jblk, mblk-1
          IF (blk < iblk) THEN           ! isrc dependency element (lower half)
            adr = iblk*bs+1; adc = blk*bs+1
          ELSE
            adr = blk*bs+1; adc = iblk*bs+1
          END IF
          !$omp task if(lut) default(shared) firstprivate(blk, jblk, iblk)             &
          !$omp   depend(inout: a(iblk*bs+1, jblk*bs+1)) depend(in: perm(blk*bs+1)) &
          !$omp   depend(in: a(blk*bs+1, jblk*bs+1)) depend(in: a(adr, adc))
          IF (.NOT. aborted) THEN
            CALL restore_if_req(iblk, jblk, blk, a, lda, bcopy, m, bs, n,      &
                                cnelim, lperm)
            CALL update_left(iblk, jblk, blk, a, lda, d, cdoff, cnelim, m, n, bs)
          END IF
          !$omp end task
        END DO
      END DO
      ! ---- update trailing (right of / at elim col) ----
      DO jblk = blk, nblk-1
        DO iblk = jblk, mblk-1
          !$omp task if(lut) default(shared) firstprivate(blk, jblk, iblk)             &
          !$omp   depend(inout: a(iblk*bs+1, jblk*bs+1)) depend(in: perm(blk*bs+1)) &
          !$omp   depend(in: a(iblk*bs+1, blk*bs+1)) depend(in: a(jblk*bs+1, blk*bs+1))
          IF (.NOT. aborted) THEN
            CALL restore_if_req(iblk, jblk, blk, a, lda, bcopy, m, bs, n,      &
                                cnelim, lperm)
            CALL update_right(iblk, jblk, blk, a, lda, d, cdoff, cnelim, m, n, bs)
          END IF
          !$omp end task
        END DO
      END DO
    END DO
    !$omp end taskgroup
  END SUBROUTINE run_pivoted

  INTEGER(ip_) FUNCTION blk_ncol(blk, n, bs) RESULT(r)
    INTEGER(ip_), INTENT(IN) :: blk, n, bs
    r = MIN(bs, n - blk*bs)
  END FUNCTION blk_ncol

  INTEGER(ip_) FUNCTION blk_nrow(blk, m, bs) RESULT(r)
    INTEGER(ip_), INTENT(IN) :: blk, m, bs
    r = MIN(bs, m - blk*bs)
  END FUNCTION blk_nrow

  SUBROUTINE permute_blkperm(perm, blk, bs, nc, lperm)
    INTEGER(ip_), INTENT(INOUT) :: perm(*)
    INTEGER(ip_), INTENT(IN)    :: blk, bs, nc, lperm(*)
    INTEGER(ip_) :: i, tmp(bs)
    DO i = 1, nc
      tmp(i) = perm(blk*bs + lperm(i) + 1)
    END DO
    DO i = 1, nc
      perm(blk*bs + i) = tmp(i)
    END DO
  END SUBROUTINE permute_blkperm

!------------------ backup / restore ------------------
  SUBROUTINE bkp_create(iblk, jblk, a, lda, bcopy, ldb, bs, m, n)
    INTEGER(ip_), INTENT(IN)    :: iblk, jblk, lda, ldb, bs, m, n
    REAL(rp_),    INTENT(IN)    :: a(lda,*)
    REAL(rp_),    INTENT(INOUT) :: bcopy(ldb,*)
    INTEGER(ip_) :: i, j, nr, nco
    nr = blk_nrow(iblk, m, bs); nco = blk_ncol(jblk, n, bs)
    DO j = 1, nco
      DO i = 1, nr
        bcopy(iblk*bs+i, jblk*bs+j) = a(iblk*bs+i, jblk*bs+j)
      END DO
    END DO
  END SUBROUTINE bkp_create

  SUBROUTINE bkp_rperm(iblk, jblk, a, lda, bcopy, ldb, bs, n, lperm, nperm)
    INTEGER(ip_), INTENT(IN)    :: iblk, jblk, lda, ldb, bs, n, nperm
    INTEGER(ip_), INTENT(IN)    :: lperm(*)
    REAL(rp_),    INTENT(INOUT) :: a(lda,*), bcopy(ldb,*)
    INTEGER(ip_) :: i, j, r, nr, nco
    nr = blk_nrow(iblk, ldb, bs)    ! ldb passed as m
    nco = blk_ncol(jblk, n, bs)
    DO j = 1, nco
      DO i = 1, nperm
        r = lperm(i) + 1
        bcopy(iblk*bs+i, jblk*bs+j) = a(iblk*bs+r, jblk*bs+j)
      END DO
      DO i = nperm+1, nr
        bcopy(iblk*bs+i, jblk*bs+j) = a(iblk*bs+i, jblk*bs+j)
      END DO
    END DO
    DO j = 1, nco
      DO i = 1, nperm
        a(iblk*bs+i, jblk*bs+j) = bcopy(iblk*bs+i, jblk*bs+j)
      END DO
    END DO
  END SUBROUTINE bkp_rperm

  SUBROUTINE bkp_cperm(iblk, jblk, a, lda, bcopy, ldb, bs, n, lperm)
    INTEGER(ip_), INTENT(IN)    :: iblk, jblk, lda, ldb, bs, n
    INTEGER(ip_), INTENT(IN)    :: lperm(*)
    REAL(rp_),    INTENT(INOUT) :: a(lda,*), bcopy(ldb,*)
    INTEGER(ip_) :: i, j, c, nr, nco
    nr = blk_nrow(iblk, ldb, bs)
    nco = blk_ncol(jblk, n, bs)
    DO j = 1, nco
      c = lperm(j) + 1
      DO i = 1, nr
        bcopy(iblk*bs+i, jblk*bs+j) = a(iblk*bs+i, jblk*bs+c)
      END DO
    END DO
    DO j = 1, nco
      DO i = 1, nr
        a(iblk*bs+i, jblk*bs+j) = bcopy(iblk*bs+i, jblk*bs+j)
      END DO
    END DO
  END SUBROUTINE bkp_cperm

  SUBROUTINE bkp_restore_part(iblk, jblk, rfrom, cfrom, a, lda, bcopy, ldb, bs, n)
    INTEGER(ip_), INTENT(IN)    :: iblk, jblk, rfrom, cfrom, lda, ldb, bs, n
    REAL(rp_),    INTENT(INOUT) :: a(lda,*)
    REAL(rp_),    INTENT(IN)    :: bcopy(ldb,*)
    INTEGER(ip_) :: i, j, nr, nco
    nr = blk_nrow(iblk, ldb, bs); nco = blk_ncol(jblk, n, bs)
    DO j = cfrom, nco-1
      DO i = rfrom, nr-1
        a(iblk*bs+i+1, jblk*bs+j+1) = bcopy(iblk*bs+i+1, jblk*bs+j+1)
      END DO
    END DO
  END SUBROUTINE bkp_restore_part

  SUBROUTINE bkp_restore_sym(iblk, jblk, from, a, lda, bcopy, ldb, bs, n, lperm)
    INTEGER(ip_), INTENT(IN)    :: iblk, jblk, from, lda, ldb, bs, n
    INTEGER(ip_), INTENT(IN)    :: lperm(*)
    REAL(rp_),    INTENT(INOUT) :: a(lda,*)
    REAL(rp_),    INTENT(IN)    :: bcopy(ldb,*)
    INTEGER(ip_) :: i, j, c, r, nr, nco
    nr = blk_nrow(iblk, ldb, bs); nco = blk_ncol(jblk, n, bs)
    DO j = from, nco-1
      c = lperm(j+1)
      DO i = from, nco-1
        r = lperm(i+1)
        IF (r > c) THEN
          a(iblk*bs+i+1, jblk*bs+j+1) = bcopy(iblk*bs+r+1, jblk*bs+c+1)
        ELSE
          a(iblk*bs+i+1, jblk*bs+j+1) = bcopy(iblk*bs+c+1, jblk*bs+r+1)
        END IF
      END DO
      DO i = nco, nr-1
        a(iblk*bs+i+1, jblk*bs+j+1) = bcopy(iblk*bs+i+1, jblk*bs+c+1)
      END DO
    END DO
  END SUBROUTINE bkp_restore_sym

  SUBROUTINE restore_if_req(iblk, jblk, elim, a, lda, bcopy, ldb, bs, n, &
                            cnelim, lperm)
    INTEGER(ip_), INTENT(IN)    :: iblk, jblk, elim, lda, ldb, bs, n
    INTEGER(ip_), INTENT(IN)    :: cnelim(0:*), lperm(bs,0:*)
    REAL(rp_),    INTENT(INOUT) :: a(lda,*)
    REAL(rp_),    INTENT(IN)    :: bcopy(ldb,*)
    INTEGER(ip_) :: rfrom
    IF (iblk == elim .AND. jblk == elim) THEN
      IF (cnelim(iblk) < blk_ncol(iblk,n,bs)) &
        CALL bkp_restore_sym(iblk, jblk, cnelim(iblk), a, lda, bcopy, ldb, bs, &
                             n, lperm(1,iblk))
    ELSE IF (iblk == elim) THEN
      IF (cnelim(iblk) < blk_nrow(iblk,ldb,bs)) &
        CALL bkp_restore_part(iblk, jblk, cnelim(iblk), cnelim(jblk), a, lda, &
                              bcopy, ldb, bs, n)
    ELSE IF (jblk == elim) THEN
      IF (cnelim(jblk) < blk_ncol(jblk,n,bs)) THEN
        rfrom = 0
        IF (iblk <= elim) rfrom = cnelim(iblk)
        CALL bkp_restore_part(iblk, jblk, rfrom, cnelim(jblk), a, lda, bcopy, &
                              ldb, bs, n)
      END IF
    END IF
  END SUBROUTINE restore_if_req

!------------------ apply_pivot_app (T / N) ------------------
  INTEGER(ip_) FUNCTION apply_T(blk, jblk, a, lda, d, cdoff, cnelim, m, n, bs, &
        small, u) RESULT(res)
    INTEGER(ip_), INTENT(IN)    :: blk, jblk, lda, m, n, bs
    INTEGER(ip_), INTENT(IN)    :: cdoff(0:*), cnelim(0:*)
    REAL(rp_),    INTENT(INOUT) :: a(lda,*)
    REAL(rp_),    INTENT(IN)    :: d(*), small, u
    INTEGER(ip_) :: nc
    nc = blk_ncol(jblk, n, bs)
    ! ApplyT: diag = block(blk,blk), aval = block(blk,jblk)
    CALL app_apply_pivot(.TRUE., cnelim(blk), nc, cnelim(jblk), &
         a(blk*bs+1, blk*bs+1), lda, d(cdoff(blk)+1), small, &
         a(blk*bs+1, jblk*bs+1), lda)
    res = app_check_threshold(.TRUE., 0_ip_, cnelim(blk), cnelim(jblk), nc, u, &
         a(blk*bs+1, jblk*bs+1), lda)
  END FUNCTION apply_T

  INTEGER(ip_) FUNCTION apply_N(iblk, blk, a, lda, d, cdoff, cnelim, m, n, bs, &
        small, u) RESULT(res)
    INTEGER(ip_), INTENT(IN)    :: iblk, blk, lda, m, n, bs
    INTEGER(ip_), INTENT(IN)    :: cdoff(0:*), cnelim(0:*)
    REAL(rp_),    INTENT(INOUT) :: a(lda,*)
    REAL(rp_),    INTENT(IN)    :: d(*), small, u
    INTEGER(ip_) :: nr
    nr = blk_nrow(iblk, m, bs)
    ! ApplyN: diag = block(blk,blk), aval = block(iblk,blk)
    CALL app_apply_pivot(.FALSE., nr, cnelim(blk), 0_ip_, &
         a(blk*bs+1, blk*bs+1), lda, d(cdoff(blk)+1), small, &
         a(iblk*bs+1, blk*bs+1), lda)
    res = app_check_threshold(.FALSE., 0_ip_, nr, 0_ip_, cnelim(blk), u, &
         a(iblk*bs+1, blk*bs+1), lda)
  END FUNCTION apply_N

!------------------ update (right / left) ------------------
  SUBROUTINE update_right(iblk, jblk, blk, a, lda, d, cdoff, cnelim, m, n, bs)
    INTEGER(ip_), INTENT(IN)    :: iblk, jblk, blk, lda, m, n, bs
    INTEGER(ip_), INTENT(IN)    :: cdoff(0:*), cnelim(0:*)
    REAL(rp_),    INTENT(INOUT) :: a(lda,*)
    REAL(rp_),    INTENT(IN)    :: d(*)
    INTEGER(ip_) :: rfrom, cfrom, nr, nco, ke, ldld
    REAL(rp_), ALLOCATABLE :: ld(:,:)
    ke = cnelim(blk)
    IF (ke == 0) RETURN
    nr = blk_nrow(iblk, m, bs); nco = blk_ncol(jblk, n, bs)
    rfrom = 0; IF (iblk <= blk) rfrom = cnelim(iblk)
    cfrom = 0; IF (jblk <= blk) cfrom = cnelim(jblk)
    IF (nr-rfrom <= 0 .OR. nco-cfrom <= 0) RETURN
    ldld = nr
    ALLOCATE(ld(MAX(nr,1), MAX(ke,1)))
    ! isrc = (iblk,blk) rows [rfrom,nr); calcLD OP_N
    CALL calc_ld(.FALSE., nr-rfrom, ke, a(iblk*bs+rfrom+1, blk*bs+1), lda, &
                 d(cdoff(blk)+1), ld(rfrom+1,1), ldld)
    ! jsrc = (jblk,blk) rows [cfrom,nco)
    CALL DGEMM('N', 'T', nr-rfrom, nco-cfrom, ke, -1.0_rp_, ld(rfrom+1,1), ldld, &
               a(jblk*bs+cfrom+1, blk*bs+1), lda, 1.0_rp_, &
               a(iblk*bs+rfrom+1, jblk*bs+cfrom+1), lda)
    DEALLOCATE(ld)
  END SUBROUTINE update_right

  SUBROUTINE update_left(iblk, jblk, blk, a, lda, d, cdoff, cnelim, m, n, bs)
    INTEGER(ip_), INTENT(IN)    :: iblk, jblk, blk, lda, m, n, bs
    INTEGER(ip_), INTENT(IN)    :: cdoff(0:*), cnelim(0:*)
    REAL(rp_),    INTENT(INOUT) :: a(lda,*)
    REAL(rp_),    INTENT(IN)    :: d(*)
    INTEGER(ip_) :: rfrom, cfrom, nr, nco, ke, ldld, isrc_r, isrc_c
    REAL(rp_), ALLOCATABLE :: ld(:,:)
    ke = cnelim(blk)
    IF (ke == 0) RETURN
    nr = blk_nrow(iblk, m, bs); nco = blk_ncol(jblk, n, bs)
    rfrom = 0; IF (iblk <= blk) rfrom = cnelim(iblk)
    cfrom = 0; IF (jblk <= blk) cfrom = cnelim(jblk)
    IF (nr-rfrom <= 0 .OR. nco-cfrom <= 0) RETURN
    ldld = nr
    ALLOCATE(ld(MAX(nr,1), MAX(ke,1)))
    IF (blk <= iblk) THEN
      isrc_r = iblk; isrc_c = blk
      ! isrc.j_ == elim_col -> calcLD OP_N, isrc.aval_[rfrom] (row offset)
      CALL calc_ld(.FALSE., nr-rfrom, ke, a(isrc_r*bs+rfrom+1, isrc_c*bs+1), &
                   lda, d(cdoff(blk)+1), ld(rfrom+1,1), ldld)
    ELSE
      isrc_r = blk; isrc_c = iblk
      ! isrc.j_ != elim_col -> calcLD OP_T, isrc.aval_[rfrom*lda] (col offset)
      CALL calc_ld(.TRUE., nr-rfrom, ke, a(isrc_r*bs+1, isrc_c*bs+rfrom+1), &
                   lda, d(cdoff(blk)+1), ld(rfrom+1,1), ldld)
    END IF
    ! jsrc = (blk,jblk), col offset cfrom
    CALL DGEMM('N', 'N', nr-rfrom, nco-cfrom, ke, -1.0_rp_, ld(rfrom+1,1), ldld, &
               a(blk*bs+1, jblk*bs+cfrom+1), lda, 1.0_rp_, &
               a(iblk*bs+rfrom+1, jblk*bs+cfrom+1), lda)
    DEALLOCATE(ld)
  END SUBROUTINE update_left

!------------------ compaction: move_back / copy_failed / move_up ------
  SUBROUTINE move_back(nc, nelim, perm, elim_perm, failed_perm)
    INTEGER(ip_), INTENT(IN)    :: nc, nelim
    INTEGER(ip_), INTENT(IN)    :: perm(*)
    INTEGER(ip_), INTENT(INOUT) :: elim_perm(*), failed_perm(*)
    INTEGER(ip_) :: i
    ! perm and elim_perm may overlap (elim_perm points earlier into same array)
    DO i = 1, nelim
      elim_perm(i) = perm(i)
    END DO
    DO i = nelim+1, nc
      failed_perm(i-nelim) = perm(i)
    END DO
  END SUBROUTINE move_back

  SUBROUTINE copy_failed_diag(iblk, jblk, mib, njb, inelim, jnelim, a, lda, &
        fdiag, ldc, num_elim, nfail, jins, ifl, iins, jf, bs)
    INTEGER(ip_), INTENT(IN)    :: iblk, jblk, mib, njb, inelim, jnelim, lda, ldc
    INTEGER(ip_), INTENT(IN)    :: num_elim, nfail, jins, ifl, iins, jf, bs
    REAL(rp_),    INTENT(IN)    :: a(lda,*)
    REAL(rp_),    INTENT(INOUT) :: fdiag(ldc,*)
    INTEGER(ip_) :: i, j, iout, jout, r0, c0
    r0 = iblk*bs; c0 = jblk*bs
    ! rows: failed rows (i>=inelim), elim cols (j<jnelim) -> rout @ (jins col, ifl row)
    DO j = 0, jnelim-1
      iout = 0
      DO i = inelim, mib-1
        fdiag(ifl+iout+1, jins+j+1) = a(r0+i+1, c0+j+1)
        iout = iout + 1
      END DO
    END DO
    ! cols^T (only if off-diagonal block): elim rows (i<inelim), failed cols (j>=jnelim)
    IF (iblk /= jblk) THEN
      jout = 0
      DO j = jnelim, njb-1
        DO i = 0, inelim-1
          fdiag(jf+jout+1, iins+i+1) = a(r0+i+1, c0+j+1)
        END DO
        jout = jout + 1
      END DO
    END IF
    ! failed x failed intersection -> dout @ (num_elim+jf col, ifl row)
    jout = 0
    DO j = jnelim, njb-1
      iout = 0
      DO i = inelim, mib-1
        fdiag(ifl+iout+1, num_elim+jf+jout+1) = a(r0+i+1, c0+j+1)
        iout = iout + 1
      END DO
      jout = jout + 1
    END DO
  END SUBROUTINE copy_failed_diag

  SUBROUTINE copy_failed_rect(jblk, jnelim, njb, a, lda, frect, ldr, jf, n, m, bs)
    INTEGER(ip_), INTENT(IN)    :: jblk, jnelim, njb, lda, ldr, jf, n, m, bs
    REAL(rp_),    INTENT(IN)    :: a(lda,*)
    REAL(rp_),    INTENT(INOUT) :: frect(ldr,*)
    INTEGER(ip_) :: j, i, jout, c0
    c0 = jblk*bs
    jout = 0
    DO j = jnelim, njb-1
      DO i = 0, m-n-1
        frect(i+1, jf+jout+1) = a(n+i+1, c0+j+1)
      END DO
      jout = jout + 1
    END DO
  END SUBROUTINE copy_failed_rect

  SUBROUTINE move_up_diag(iblk, jblk, inelim, jnelim, a, lda, iins, jins, bs)
    INTEGER(ip_), INTENT(IN)    :: iblk, jblk, inelim, jnelim, lda, iins, jins, bs
    REAL(rp_),    INTENT(INOUT) :: a(lda,*)
    INTEGER(ip_) :: i, j, r0, c0
    r0 = iblk*bs; c0 = jblk*bs
    IF (iins == r0 .AND. jins == c0) RETURN
    DO j = 0, jnelim-1
      DO i = 0, inelim-1
        a(iins+i+1, jins+j+1) = a(r0+i+1, c0+j+1)
      END DO
    END DO
  END SUBROUTINE move_up_diag

  SUBROUTINE move_up_rect_all(jblk, jnelim, a, lda, jins, n, m, bs)
    INTEGER(ip_), INTENT(IN)    :: jblk, jnelim, lda, jins, n, m, bs
    REAL(rp_),    INTENT(INOUT) :: a(lda,*)
    INTEGER(ip_) :: i, j, c0
    c0 = jblk*bs
    IF (jins == c0) RETURN
    DO j = 0, jnelim-1
      DO i = n, m-1
        a(i+1, jins+j+1) = a(i+1, c0+j+1)
      END DO
    END DO
  END SUBROUTINE move_up_rect_all

!====================== unpivoted (aggressive) path ====================
  ! Apply the diagonal block's row permutation lperm to the first ncol(iblk)
  ! rows of block (iblk,jblk), no backup (used in the optimistic unpivoted pass).
  SUBROUTINE apply_rperm(iblk, jblk, a, lda, bs, m, n, lperm)
    INTEGER(ip_), INTENT(IN)    :: iblk, jblk, lda, bs, m, n, lperm(*)
    REAL(rp_),    INTENT(INOUT) :: a(lda,*)
    INTEGER(ip_) :: j, i, r, nco, np
    REAL(rp_), ALLOCATABLE :: lw(:,:)
    nco = blk_ncol(jblk, n, bs); np = blk_ncol(iblk, n, bs)
    ALLOCATE(lw(MAX(np,1), MAX(nco,1)))
    DO j = 1, nco
      DO i = 1, np
        r = lperm(i) + 1
        lw(i, j) = a(iblk*bs+r, jblk*bs+j)
      END DO
    END DO
    DO j = 1, nco
      DO i = 1, np
        a(iblk*bs+i, jblk*bs+j) = lw(i, j)
      END DO
    END DO
    DEALLOCATE(lw)
  END SUBROUTINE apply_rperm

  ! Apply the diagonal block's column permutation lperm to block (iblk,jblk).
  SUBROUTINE apply_cperm(iblk, jblk, a, lda, bs, m, n, lperm)
    INTEGER(ip_), INTENT(IN)    :: iblk, jblk, lda, bs, m, n, lperm(*)
    REAL(rp_),    INTENT(INOUT) :: a(lda,*)
    INTEGER(ip_) :: j, i, c, nco, nr
    REAL(rp_), ALLOCATABLE :: lw(:,:)
    nco = blk_ncol(jblk, n, bs); nr = blk_nrow(iblk, m, bs)
    ALLOCATE(lw(MAX(nr,1), MAX(nco,1)))
    DO j = 1, nco
      c = lperm(j) + 1
      DO i = 1, nr
        lw(i, j) = a(iblk*bs+i, jblk*bs+c)
      END DO
    END DO
    DO j = 1, nco
      DO i = 1, nr
        a(iblk*bs+i, jblk*bs+j) = lw(i, j)
      END DO
    END DO
    DEALLOCATE(lw)
  END SUBROUTINE apply_cperm

  ! Inverse of apply_rperm (undo a failed row permutation on recovery).
  SUBROUTINE apply_inv_rperm(iblk, jblk, a, lda, bs, m, n, lperm)
    INTEGER(ip_), INTENT(IN)    :: iblk, jblk, lda, bs, m, n, lperm(*)
    REAL(rp_),    INTENT(INOUT) :: a(lda,*)
    INTEGER(ip_) :: j, i, r, nco, np
    REAL(rp_), ALLOCATABLE :: lw(:,:)
    nco = blk_ncol(jblk, n, bs); np = blk_ncol(iblk, n, bs)
    ALLOCATE(lw(MAX(np,1), MAX(nco,1)))
    DO j = 1, nco
      DO i = 1, np
        r = lperm(i) + 1
        lw(r, j) = a(iblk*bs+i, jblk*bs+j)
      END DO
    END DO
    DO j = 1, nco
      DO i = 1, np
        a(iblk*bs+i, jblk*bs+j) = lw(i, j)
      END DO
    END DO
    DEALLOCATE(lw)
  END SUBROUTINE apply_inv_rperm

  ! Number of columns in the accepted leading prefix of fully-passed block cols.
  INTEGER(ip_) FUNCTION calc_nelim_up(m, bs, mblk, nblk, cnpass, cnelim) RESULT(res)
    INTEGER(ip_), INTENT(IN) :: m, bs, mblk, nblk, cnpass(0:*), cnelim(0:*)
    INTEGER(ip_) :: j
    res = 0
    DO j = 0, nblk-1
      IF (cnpass(j) == mblk - j) THEN
        res = res + cnelim(j)
      ELSE
        EXIT
      END IF
    END DO
  END FUNCTION calc_nelim_up

  ! Optimistic unpivoted factorization (port of run_elim_unpivoted_notasks):
  ! assume every pivot passes; abort at the first block column that does not
  ! fully eliminate or whose below-block fails the a-posteriori test. Records
  ! per-block progress in up2d for restore(). Serial. A full backup of a into
  ! bcopy must be taken by the caller beforehand.
  RECURSIVE SUBROUTINE run_unpivoted(m, n, perm, a, lda, d, u, small, action,  &
        bs, inner, aggr, mblk, nblk, cnelim, cnpass, cfirst, cdoff, lperm, up2d,&
        lut, num_elim, flag)
    INTEGER(ip_), INTENT(IN)    :: m, n, lda, bs, inner, mblk, nblk
    INTEGER(ip_), INTENT(INOUT) :: perm(*)
    REAL(rp_),    INTENT(INOUT) :: a(lda,*), d(*)
    REAL(rp_),    INTENT(IN)    :: u, small
    LOGICAL,     INTENT(IN)    :: action, aggr, lut
    INTEGER(ip_), INTENT(INOUT) :: cnelim(0:*), cnpass(0:*), cdoff(0:*)
    LOGICAL,     INTENT(INOUT) :: cfirst(0:*)
    INTEGER(ip_), INTENT(INOUT) :: lperm(bs, 0:*), up2d(0:mblk-1, 0:nblk-1)
    INTEGER(ip_), INTENT(OUT)   :: num_elim, flag
    INTEGER(ip_) :: blk, jblk, iblk, next_elim, i
    INTEGER(ip_) :: tnc, tnr, tdoff, tnelim, tbp, tlflag
    LOGICAL     :: aborted, la
    REAL(rp_)    :: adum(1,1)
    REAL(rp_), ALLOCATABLE :: tldw(:,:)
    ! Task-parallel optimistic unpivoted pass (port of run_elim_unpivoted). The
    ! full backup lives in bcopy (taken by the caller), so tasks never back up.
    ! Dependencies are purely on the a-blocks (no adjust / no perm token); each
    ! task records its progress in up2d and bails out if another task has already
    ! signalled abort. With lut=.false. (or no OpenMP) every task is undeferred
    ! and this runs in the exact serial order of run_elim_unpivoted_notasks.
    flag = 0; next_elim = 0; aborted = .FALSE.
    !$omp taskgroup
    DO blk = 0, nblk-1
      ! --- factor diagonal block ---
      !$omp task if(lut) default(shared) firstprivate(blk)                      &
      !$omp   private(tnc, tnr, tdoff, tnelim, tldw, tlflag, i, la)             &
      !$omp   depend(inout: a(blk*bs+1, blk*bs+1))
      !$omp atomic read
      la = aborted
      IF (.NOT. la) THEN
        tnc = blk_ncol(blk, n, bs); tnr = blk_nrow(blk, m, bs)
        tdoff = 2*next_elim; cdoff(blk) = tdoff
        DO i = 1, tnc
          lperm(i, blk) = i-1
        END DO
        up2d(blk, blk) = blk
        tlflag = 0
        IF (bs > inner) THEN
          tnelim = ldlt_app_factor(tnr, tnc, lperm(1,blk), a(blk*bs+1,blk*bs+1),&
                     lda, d(tdoff+1), u, small, action, inner, tlflag, .FALSE., &
                     aggr)
          IF (tlflag == 0) CALL permute_blkperm(perm, blk, bs, tnc, lperm(1,blk))
        ELSE IF (tnc < bs) THEN
          ALLOCATE(tldw(tnr+2, 2))
          tnelim = ldlt_tpp_factor(tnr, tnc, lperm(1,blk), a(blk*bs+1,blk*bs+1),&
                     lda, d(tdoff+1), tldw, tnr+2, action, u, small, 0_ip_, adum,&
                     1_ip_, tlflag)
          DEALLOCATE(tldw)
          IF (tlflag == 0) CALL permute_blkperm(perm, blk, bs, tnc, lperm(1,blk))
        ELSE
          ALLOCATE(tldw(bs, bs))
          CALL block_ldlt(0_ip_, perm(blk*bs+1), a(blk*bs+1,blk*bs+1), lda,      &
                          d(tdoff+1), tldw, action, u, small, lperm(1,blk), bs, &
                          tlflag)
          DEALLOCATE(tldw); tnelim = bs
        END IF
        IF (tlflag /= 0) THEN
          flag = MERGE(FLAG_OOM, FLAG_SINGULAR, tlflag == FLAG_OOM)   ! keep OOM(2) vs abort(1)
          !$omp atomic write
          aborted = .TRUE.
        ELSE
          cnelim(blk) = tnelim
          IF (tnelim < tnc) THEN
            cnpass(blk) = 0                 ! diagonal not fully eliminated
            !$omp atomic write
            aborted = .TRUE.
          ELSE
            cfirst(blk) = (blk == 0)
            cnpass(blk) = 1                 ! init_passed(1): diagonal passed
            next_elim = next_elim + tnelim
          END IF
        END IF
      END IF
      !$omp end task
      ! --- apply row perm to eliminated ROW blocks (jblk<blk) ---
      DO jblk = 0, blk-1
        !$omp task if(lut) default(shared) firstprivate(blk, jblk) private(la)  &
        !$omp   depend(in: a(blk*bs+1, blk*bs+1))                               &
        !$omp   depend(inout: a(blk*bs+1, jblk*bs+1))
        !$omp atomic read
        la = aborted
        IF (.NOT. la) THEN
          up2d(blk, jblk) = blk
          CALL apply_rperm(blk, jblk, a, lda, bs, m, n, lperm(1,blk))
        END IF
        !$omp end task
      END DO
      ! --- apply col perm + pivot to below blocks (iblk>blk), test threshold ---
      DO iblk = blk+1, mblk-1
        !$omp task if(lut) default(shared) firstprivate(blk, iblk) private(tbp, la) &
        !$omp   depend(in: a(blk*bs+1, blk*bs+1))                               &
        !$omp   depend(inout: a(iblk*bs+1, blk*bs+1))
        !$omp atomic read
        la = aborted
        IF (.NOT. la) THEN
          up2d(iblk, blk) = blk
          CALL apply_cperm(iblk, blk, a, lda, bs, m, n, lperm(1,blk))
          tbp = apply_N(iblk, blk, a, lda, d, cdoff, cnelim, m, n, bs, small, u)
          IF (tbp < cnelim(blk)) THEN       ! test_fail -> column not fully passed
            !$omp atomic write
            aborted = .TRUE.
          ELSE
            !$omp atomic update
            cnpass(blk) = cnpass(blk) + 1
          END IF
        END IF
        !$omp end task
      END DO
      ! --- update trailing columns [blk+1, nblk) (optimistic, no restore) ---
      DO jblk = blk+1, nblk-1
        DO iblk = jblk, mblk-1
          !$omp task if(lut) default(shared) firstprivate(blk, jblk, iblk) private(la) &
          !$omp   depend(inout: a(iblk*bs+1, jblk*bs+1))                        &
          !$omp   depend(in: a(iblk*bs+1, blk*bs+1)) depend(in: a(jblk*bs+1, blk*bs+1))
          !$omp atomic read
          la = aborted
          IF (.NOT. la) THEN
            up2d(iblk, jblk) = blk
            CALL update_right(iblk, jblk, blk, a, lda, d, cdoff, cnelim, m, n, bs)
          END IF
          !$omp end task
        END DO
      END DO
    END DO
    !$omp end taskgroup
    IF (flag /= 0) THEN
      num_elim = -1
    ELSE
      num_elim = calc_nelim_up(m, bs, mblk, nblk, cnpass, cnelim)
    END IF
  END SUBROUTINE run_unpivoted

  ! Roll back the matrix after a failed optimistic pass to a state consistent
  ! with nelim_blk accepted block columns, ready for the careful pivoted pass to
  ! resume from block nelim_blk (port of restore(), serial).
  SUBROUTINE restore_unpiv(nelim_blk, m, n, perm, a, lda, d, bs, mblk, nblk,    &
        cnelim, cdoff, lperm, bcopy, up2d, old_perm)
    INTEGER(ip_), INTENT(IN)    :: nelim_blk, m, n, lda, bs, mblk, nblk
    INTEGER(ip_), INTENT(INOUT) :: perm(*)
    REAL(rp_),    INTENT(INOUT) :: a(lda,*)
    REAL(rp_),    INTENT(IN)    :: d(*), bcopy(m,*)
    INTEGER(ip_), INTENT(IN)    :: cnelim(0:*), cdoff(0:*), lperm(bs,0:*)
    INTEGER(ip_), INTENT(IN)    :: up2d(0:mblk-1, 0:nblk-1), old_perm(*)
    INTEGER(ip_) :: i, jblk, iblk, kblk, progress
    ! 1. restore permutation of the failed part
    DO i = nelim_blk*bs, n-1
      perm(i+1) = old_perm(i+1)
    END DO
    ! 2. undo failed row perms in accepted columns
    DO jblk = 0, nelim_blk-1
      DO iblk = nelim_blk, nblk-1
        IF (up2d(iblk, jblk) >= nelim_blk) &
          CALL apply_inv_rperm(iblk, jblk, a, lda, bs, m, n, lperm(1,iblk))
      END DO
    END DO
    ! 3. failed columns: full reset of over-updated blocks + apply missing updates
    DO jblk = nelim_blk, nblk-1
      DO iblk = jblk, mblk-1
        progress = up2d(iblk, jblk)
        IF (progress >= nelim_blk) THEN
          CALL bkp_restore_part(iblk, jblk, 0_ip_, 0_ip_, a, lda, bcopy, m, bs, n)
          progress = -1
        END IF
        DO kblk = progress+1, nelim_blk-1
          CALL update_right(iblk, jblk, kblk, a, lda, d, cdoff, cnelim, m, n, bs)
        END DO
      END DO
    END DO
  END SUBROUTINE restore_unpiv




! ========================= factor_node_indef ==========================
   SUBROUTINE factor_node_indef(nrow, ncol, ndelay_in, a, lda, d, perm, &
         contrib, ldcontrib, action, u, small, nb, posdef, nelim, ndelay_out, &
         failed_tpp, nfirst, nsecond, alloc_err)
      INTEGER(ip_), INTENT(IN)    :: nrow, ncol, ndelay_in, lda, ldcontrib, nb
      REAL(rp_),    INTENT(INOUT) :: a(lda, *), d(*)
      INTEGER(ip_), INTENT(INOUT) :: perm(*)
      REAL(rp_),    INTENT(INOUT) :: contrib(ldcontrib, *)
      LOGICAL,       INTENT(IN)    :: action, posdef
      REAL(rp_),    INTENT(IN)    :: u, small
      INTEGER(ip_), INTENT(OUT)   :: nelim, ndelay_out
      LOGICAL,       INTENT(IN), OPTIONAL :: failed_tpp
      ! columns not eliminated by the 1st (APP) / 2nd (TPP finish) pass, for stats
      INTEGER(ip_), INTENT(OUT), OPTIONAL :: nfirst, nsecond
      LOGICAL,       INTENT(OUT), OPTIONAL :: alloc_err   ! .true. on OOM
      INTEGER(ip_) :: m, n, flag, nbe, nelim2, ldld, nelim_app, st
      LOGICAL :: ftpp, lfin
      REAL(rp_), ALLOCATABLE :: ldblk(:, :), ldw(:, :)
      m = nrow + ndelay_in
      n = ncol + ndelay_in
      IF (PRESENT(nfirst))   nfirst   = 0
      IF (PRESENT(nsecond))  nsecond  = 0
      IF (PRESENT(alloc_err)) alloc_err = .FALSE.
      IF (posdef) THEN
         ! Cholesky path (no pivoting, no delays); ndelay_in is always 0 here
         CALL chol_factor_node(m, n, a, lda, d, contrib, ldcontrib, flag)
         IF (flag == -2) THEN        ! out of memory in the Cholesky contrib
            IF (PRESENT(alloc_err)) alloc_err = .TRUE.
            nelim = -1
         ELSE IF (flag < 0) THEN
            nelim = -1               ! not positive definite (DPOTRF failed)
         ELSE
            nelim = n
         END IF
         ndelay_out = 0
         RETURN
      END IF
      ! failed_pivot_method: TPP (default) retries failed columns with ldlt_tpp
      ftpp = .TRUE.; IF (PRESENT(failed_tpp)) ftpp = failed_tpp
      nbe = nb
      IF (nbe == 0) THEN
         ! TPP method (sentinel nb=0): single unblocked ldlt_tpp panel
         nelim = ldlt_blocked_factor(m, n, perm, a, lda, d, u, small, action, &
                                     n, flag)
      ELSE IF (nbe > 0) THEN
         ! APP_BLOCK: a-posteriori pivoted, outer block nbe (= block_size),
         ! inner block 32 via recursion, wide BLAS-3 updates.
         nelim = ldlt_app_factor(m, n, perm, a, lda, d, u, small, action, nbe, &
                                 flag)
      ELSE
         ! APP_AGGRESSIVE (nb<0): optimistic unpivoted-first at |nb|, falling
         ! back to the pivoted pass on any a-posteriori failure.
         nelim = ldlt_app_factor(m, n, perm, a, lda, d, u, small, action, -nbe, &
                                 flag, aggressive = .TRUE.)
      END IF
      IF (flag == FLAG_OOM) THEN            ! out of memory inside ldlt_app_factor
         IF (PRESENT(alloc_err)) alloc_err = .TRUE.
         nelim = -1; ndelay_out = 0; RETURN
      END IF
      IF (nelim < 0 .OR. flag < 0) THEN  ! singular pivot, action=.FALSE. -> abort
         nelim = -1; ndelay_out = 0; RETURN   ! caller maps nelim<0 to ERROR_SINGULAR
      END IF
      ! Finish off any APP-failed columns with TPP (port of the failed_pivot_method
      ! branch in cpu/factor.hxx): always at a root (m==n, no parent to delay to),
      ! and at every node when failed_pivot_method = TPP (the default). This
      ! reduces the number of delayed pivots and avoids spurious singularity.
      nelim_app = nelim                       ! nelim after the first (APP) pass
      lfin = .FALSE.
      IF (nbe /= 0 .AND. nelim < n .AND. (m == n .OR. ftpp)) THEN
         ldld = m - nelim + 2
         ALLOCATE(ldw(ldld, 2), stat=st)
         IF (st /= 0) THEN
            IF (PRESENT(alloc_err)) alloc_err = .TRUE.
            nelim = -1; ndelay_out = 0; RETURN
         END IF
         nelim2 = ldlt_tpp_factor(m-nelim, n-nelim, perm(nelim+1), &
                     a(nelim+1, nelim+1), lda, d(2*nelim+1), ldw, ldld, action, &
                     u, small, nelim, a(nelim+1, 1), lda, flag)
         DEALLOCATE(ldw)
         IF (flag < 0) THEN            ! singular during TPP finish, action=.FALSE.
            nelim = -1; ndelay_out = 0; RETURN
         END IF
         nelim = nelim + nelim2
         lfin = .TRUE.
      END IF
      ! not_first_pass / not_second_pass (as in cpu/factor.hxx): for TPP the tpp
      ! IS the first pass; for APP the first pass is the a-posteriori one and the
      ! (optional) TPP finish is the second.
      IF (PRESENT(nfirst)) THEN
         IF (nbe == 0) THEN
            nfirst = n - nelim
         ELSE
            nfirst = n - nelim_app
         END IF
      END IF
      IF (PRESENT(nsecond)) THEN
         IF (nbe /= 0 .AND. lfin) nsecond = n - nelim
      END IF
      IF (m-n > 0 .AND. nelim > 0) THEN
         ALLOCATE(ldblk(m-n, nelim), stat=st)
         IF (st /= 0) THEN
            IF (PRESENT(alloc_err)) alloc_err = .TRUE.
            nelim = -1; ndelay_out = 0; RETURN
         END IF
         CALL calc_ld(.FALSE., m-n, nelim, a(n+1, 1), lda, d, ldblk, m-n)
         CALL DGEMM('N', 'T', m-n, m-n, nelim, -1.0_rp_, a(n+1, 1), lda, &
                    ldblk, m-n, 0.0_rp_, contrib, ldcontrib)
         DEALLOCATE(ldblk)
      END IF
      ndelay_out = n - nelim
   END SUBROUTINE factor_node_indef

   !> Cholesky (LL^T) node factor for positive-definite fronts, stored in the
   !! same unit-L + D^-1 layout as the indef path (L_chol column j divided by its
   !! diagonal, D_j = diag_j^2) so the assembly, solves and enquire are reused
   !! unchanged. No pivoting / no delays. flag<0 if the block is not SPD.
   SUBROUTINE chol_factor_node(m, n, a, lda, d, contrib, ldcontrib, flag)
      INTEGER(ip_), INTENT(IN)    :: m, n, lda, ldcontrib
      REAL(rp_),    INTENT(INOUT) :: a(lda, *), d(*)
      REAL(rp_),    INTENT(INOUT) :: contrib(ldcontrib, *)
      INTEGER(ip_), INTENT(OUT)   :: flag
      INTEGER(ip_) :: j, i, info, m2
      REAL(rp_) :: dj
      REAL(rp_), ALLOCATABLE :: ldblk(:, :)
      flag = 0
      CALL DPOTRF('L', n, a, lda, info)
      IF (info /= 0) THEN; flag = FLAG_SINGULAR; RETURN; END IF
      m2 = m - n
      IF (m2 > 0) &
         CALL DTRSM('R', 'L', 'T', 'N', m2, n, 1.0_rp_, a, lda, a(n+1,1), lda)
      DO j = 1, n
         dj = a(j, j)
         d(2*j-1) = 1.0_rp_/(dj*dj); d(2*j) = 0.0_rp_
         DO i = j, m
            a(i, j) = a(i, j)/dj
         END DO
      END DO
      IF (m2 > 0) THEN
         ALLOCATE(ldblk(m2, n), stat=info)
         IF (info /= 0) THEN; flag = FLAG_OOM; RETURN; END IF   ! out of memory
         CALL calc_ld(.FALSE., m2, n, a(n+1, 1), lda, d, ldblk, m2)
         CALL DGEMM('N', 'T', m2, m2, n, -1.0_rp_, a(n+1, 1), lda, ldblk, m2, &
                    0.0_rp_, contrib, ldcontrib)
         DEALLOCATE(ldblk)
      END IF
   END SUBROUTINE chol_factor_node

! ============================ assemble ================================
   SUBROUTINE assemble_expected(from, to, cm, cache, contrib, ldc, &
                                lcol, ldl, parent_ncol)
      INTEGER(ip_), INTENT(IN)    :: from, to, cm, ldc, ldl, parent_ncol
      INTEGER(ip_), INTENT(IN)    :: cache(*)
      REAL(rp_),    INTENT(IN)    :: contrib(ldc, *)
      REAL(rp_),    INTENT(INOUT) :: lcol(ldl, *)
      INTEGER(ip_) :: i, j, c
      DO i = from, to
         c = cache(i)
         IF (c <= parent_ncol) THEN
            DO j = i, cm
               lcol(cache(j), c) = lcol(cache(j), c) + contrib(j, i)
            END DO
         END IF
      END DO
   END SUBROUTINE assemble_expected

   SUBROUTINE assemble_expected_contrib(from, to, cm, cache, contrib, ldc, &
                                        pcontrib, ldp)
      INTEGER(ip_), INTENT(IN)    :: from, to, cm, ldc, ldp
      INTEGER(ip_), INTENT(IN)    :: cache(*)
      REAL(rp_),    INTENT(IN)    :: contrib(ldc, *)
      REAL(rp_),    INTENT(INOUT) :: pcontrib(ldp, *)
      INTEGER(ip_) :: i, j, c
      DO i = from, to
         c = cache(i)
         IF (c >= 1) THEN
            DO j = i, cm
               IF (cache(j) >= 1) &
                  pcontrib(cache(j), c) = pcontrib(cache(j), c) + contrib(j, i)
            END DO
         END IF
      END DO
   END SUBROUTINE assemble_expected_contrib

! ========================= ldlt_app solves ============================
   SUBROUTINE ldlt_app_solve_fwd(m, n, l, ldl, nrhs, x, ldx)
      INTEGER(ip_), INTENT(IN)    :: m, n, ldl, nrhs, ldx
      REAL(rp_),    INTENT(IN)    :: l(ldl, *)
      REAL(rp_),    INTENT(INOUT) :: x(ldx, *)
      IF (n <= 0) RETURN
      IF (nrhs == 1) THEN
         CALL DTRSV('L', 'N', 'U', n, l(1,1), ldl, x(1,1), 1_ip_)
         IF (m > n) &
            CALL DGEMV('N', m-n, n, -1.0_rp_, l(n+1,1), ldl, x(1,1), 1_ip_, &
                       1.0_rp_, x(n+1,1), 1_ip_)
      ELSE
         CALL DTRSM('L', 'L', 'N', 'U', n, nrhs, 1.0_rp_, l(1,1), ldl, x(1,1), ldx)
         IF (m > n) &
            CALL DGEMM('N', 'N', m-n, nrhs, n, -1.0_rp_, l(n+1,1), ldl, &
                       x(1,1), ldx, 1.0_rp_, x(n+1,1), ldx)
      END IF
   END SUBROUTINE ldlt_app_solve_fwd

   SUBROUTINE ldlt_app_solve_diag(n, d, nrhs, x, ldx)
      INTEGER(ip_), INTENT(IN)    :: n, nrhs, ldx
      REAL(rp_),    INTENT(IN)    :: d(*)
      REAL(rp_),    INTENT(INOUT) :: x(ldx, *)
      INTEGER(ip_) :: i, r
      REAL(rp_)    :: d11, d21, d22, x1, x2
      i = 0
      DO WHILE (i < n)
         IF (i+1 == n .OR. IEEE_IS_FINITE(d(2*i+3))) THEN
            d11 = d(2*i+1)
            DO r = 1, nrhs
               x(i+1, r) = x(i+1, r)*d11
            END DO
            i = i + 1
         ELSE
            d11 = d(2*i+1); d21 = d(2*i+2); d22 = d(2*i+4)
            DO r = 1, nrhs
               x1 = x(i+1, r); x2 = x(i+2, r)
               x(i+1, r) = d11*x1 + d21*x2
               x(i+2, r) = d21*x1 + d22*x2
            END DO
            i = i + 2
         END IF
      END DO
   END SUBROUTINE ldlt_app_solve_diag

   SUBROUTINE ldlt_app_solve_bwd(m, n, l, ldl, nrhs, x, ldx)
      INTEGER(ip_), INTENT(IN)    :: m, n, ldl, nrhs, ldx
      REAL(rp_),    INTENT(IN)    :: l(ldl, *)
      REAL(rp_),    INTENT(INOUT) :: x(ldx, *)
      IF (n <= 0) RETURN
      IF (nrhs == 1) THEN
         IF (m > n) &
            CALL DGEMV('T', m-n, n, -1.0_rp_, l(n+1,1), ldl, x(n+1,1), 1_ip_, &
                       1.0_rp_, x(1,1), 1_ip_)
         CALL DTRSV('L', 'T', 'U', n, l(1,1), ldl, x(1,1), 1_ip_)
      ELSE
         IF (m > n) &
            CALL DGEMM('T', 'N', n, nrhs, m-n, -1.0_rp_, l(n+1,1), ldl, &
                       x(n+1,1), ldx, 1.0_rp_, x(1,1), ldx)
         CALL DTRSM('L', 'L', 'T', 'U', n, nrhs, 1.0_rp_, l(1,1), ldl, x(1,1), ldx)
      END IF
   END SUBROUTINE ldlt_app_solve_bwd

! ===================== multifrontal driver (delays) ===================
   !> Factor one node: assemble children (+ foreign contribs), factor, form the
   !! Schur contribution. Uses a per-thread scratch pmap so it is
   !! safe to call concurrently on independent nodes. node_ok is .false. only if
   !! a root fails to eliminate all its columns.
   SUBROUTINE factor_one_node(node, p, nnodes, n, action, u, small, nb, posdef, &
                              node_ok, contribs, failed_tpp, node_aok)
      TYPE(dmf_node), INTENT(INOUT) :: node(:)
      INTEGER(ip_),  INTENT(IN)    :: p, nnodes, n, nb
      LOGICAL,        INTENT(IN)    :: action, posdef
      REAL(rp_),     INTENT(IN)    :: u, small
      LOGICAL,        INTENT(OUT)   :: node_ok
      TYPE(subtree_contrib_t), INTENT(IN), OPTIONAL :: contribs(:)
      LOGICAL,        INTENT(IN), OPTIONAL :: failed_tpp
      LOGICAL,        INTENT(OUT), OPTIONAL :: node_aok  ! .false. on OOM
      LOGICAL :: ftpp, aerr
      INTEGER(ip_) :: c, i, j, s, k, cm, ncc, pcol, ccol, crow, pr, g, dcol, ndout, ci, st
      INTEGER(ip_), ALLOCATABLE :: cache(:)
      REAL(rp_) :: contrib_dummy(1,1), val
      node_ok = .TRUE.
      IF (PRESENT(node_aok)) node_aok = .TRUE.
      ftpp = .TRUE.; IF (PRESENT(failed_tpp)) ftpp = failed_tpp
      IF (.NOT. ALLOCATED(tls_pmap)) THEN
         ALLOCATE(tls_pmap(n))
      ELSE IF (SIZE(tls_pmap) < n) THEN
         DEALLOCATE(tls_pmap); ALLOCATE(tls_pmap(n))
      END IF
      ASSOCIATE (nd => node(p))
         nd%ndelay_in = 0
         DO c = 1, nnodes
            IF (node(c)%parent == p) nd%ndelay_in = nd%ndelay_in + node(c)%ndelay_out
         END DO
         IF (PRESENT(contribs) .AND. ALLOCATED(nd%contribs)) THEN
            DO k = 1, SIZE(nd%contribs)
               nd%ndelay_in = nd%ndelay_in + contribs(nd%contribs(k))%ndelay
            END DO
         END IF
         nd%ncol = nd%symb_ncol + nd%ndelay_in
         nd%nrow = nd%symb_nrow + nd%ndelay_in
         nd%ldl  = nd%nrow
         cm = nd%symb_nrow - nd%symb_ncol
         ! guard the size-proportional (per-front) allocations against OOM
         IF (ALLOCATED(nd%lcol)) DEALLOCATE(nd%lcol)
         ALLOCATE(nd%lcol(nd%ldl, nd%ncol), stat=st)
         IF (st == 0) THEN
            IF (ALLOCATED(nd%perm)) DEALLOCATE(nd%perm)
            ALLOCATE(nd%perm(nd%ncol), stat=st)
         END IF
         IF (st == 0) THEN
            IF (ALLOCATED(nd%d)) DEALLOCATE(nd%d)
            ALLOCATE(nd%d(2*nd%ncol+2), stat=st)
         END IF
         IF (st == 0 .AND. cm > 0) THEN
            IF (ALLOCATED(nd%contrib)) DEALLOCATE(nd%contrib)
            ALLOCATE(nd%contrib(cm,cm), stat=st)
         END IF
         IF (st /= 0) THEN          ! out of memory forming this front
            node_ok = .FALSE.
            IF (PRESENT(node_aok)) node_aok = .FALSE.
            RETURN
         END IF
         nd%lcol = 0._rp_
         IF (cm > 0) nd%contrib = 0._rp_
         DO i = 1, nd%symb_ncol
            nd%perm(i) = nd%rlist(i); tls_pmap(nd%rlist(i)) = i
         END DO
         DO i = nd%symb_ncol+1, nd%symb_nrow
            tls_pmap(nd%rlist(i)) = i + nd%ndelay_in
         END DO
         IF (ALLOCATED(nd%av)) THEN
            DO k = 1, SIZE(nd%av)
               i = nd%ai(k); IF (i > nd%symb_ncol) i = i + nd%ndelay_in
               nd%lcol(i, nd%aj(k)) = nd%lcol(i, nd%aj(k)) + nd%av(k)
            END DO
         END IF
         dcol = nd%symb_ncol
         DO c = 1, nnodes
            IF (node(c)%parent /= p) CYCLE
            ASSOCIATE (ch => node(c))
            DO i = 0, ch%ndelay_out-1
               pcol = dcol + 1; ccol = ch%nelim + i + 1
               nd%perm(pcol) = ch%perm(ccol)
               DO j = 0, ch%ndelay_out-1-i
                  nd%lcol(pcol+j, pcol) = ch%lcol(ccol+j, ccol)
               END DO
               DO s = ch%symb_ncol+1, ch%symb_nrow
                  g = ch%rlist(s); crow = s + ch%ndelay_in
                  val = ch%lcol(crow, ccol); pr = tls_pmap(g)
                  IF (pr <= nd%ncol) THEN
                     nd%lcol(pcol, pr) = nd%lcol(pcol, pr) + val
                  ELSE
                     nd%lcol(pr, pcol) = nd%lcol(pr, pcol) + val
                  END IF
               END DO
               dcol = dcol + 1
            END DO
            IF (ch%symb_nrow - ch%symb_ncol > 0) THEN
               CALL build_cache(ch, tls_pmap, cache)
               CALL assemble_expected(1_ip_, ch%symb_nrow-ch%symb_ncol, &
                    ch%symb_nrow-ch%symb_ncol, cache, ch%contrib, &
                    ch%symb_nrow-ch%symb_ncol, nd%lcol, nd%ldl, nd%ncol)
            END IF
            END ASSOCIATE
         END DO
         IF (PRESENT(contribs) .AND. ALLOCATED(nd%contribs)) THEN
            DO ci = 1, SIZE(nd%contribs)
               ASSOCIATE (ct => contribs(nd%contribs(ci)))
               DO i = 1, ct%ndelay
                  pcol = dcol + 1
                  nd%perm(pcol) = ct%delay_perm(i)
                  DO j = i, ct%ndelay
                     nd%lcol(pcol + (j-i), pcol) = ct%delay_val(j, i)
                  END DO
                  DO k = 1, ct%cn
                     pr = tls_pmap(ct%rlist(k)); val = ct%delay_val(ct%ndelay + k, i)
                     IF (pr <= nd%ncol) THEN
                        nd%lcol(pcol, pr) = nd%lcol(pcol, pr) + val
                     ELSE
                        nd%lcol(pr, pcol) = nd%lcol(pr, pcol) + val
                     END IF
                  END DO
                  dcol = dcol + 1
               END DO
               IF (ct%cn > 0) THEN
                  IF (ALLOCATED(cache)) DEALLOCATE(cache)
                  ALLOCATE(cache(ct%cn))
                  DO k = 1, ct%cn
                     cache(k) = tls_pmap(ct%rlist(k))
                  END DO
                  CALL assemble_expected(1_ip_, ct%cn, ct%cn, cache, ct%val, &
                       ct%cn, nd%lcol, nd%ldl, nd%ncol)
               END IF
               END ASSOCIATE
            END DO
         END IF
         ncc = cm
         aerr = .FALSE.
         IF (ncc > 0) THEN
            CALL factor_node_indef(nd%symb_nrow, nd%symb_ncol, nd%ndelay_in, &
                 nd%lcol, nd%ldl, nd%d, nd%perm, nd%contrib, ncc, action, u, &
                 small, nb, posdef, nd%nelim, ndout, failed_tpp = ftpp, &
                 nfirst = nd%nfirst, nsecond = nd%nsecond, alloc_err = aerr)
         ELSE
            CALL factor_node_indef(nd%symb_nrow, nd%symb_ncol, nd%ndelay_in, &
                 nd%lcol, nd%ldl, nd%d, nd%perm, contrib_dummy, 1_ip_, action, &
                 u, small, nb, posdef, nd%nelim, ndout, failed_tpp = ftpp, &
                 nfirst = nd%nfirst, nsecond = nd%nsecond, alloc_err = aerr)
         END IF
         IF (aerr) THEN               ! out of memory during this node's factor
            node_ok = .FALSE.
            IF (PRESENT(node_aok)) node_aok = .FALSE.
            RETURN
         END IF
         IF (nd%nelim < 0) THEN      ! non-SPD (posdef) or singular (indef, action=F)
            node_ok = .FALSE.        ! -> ERROR_NOT_POS_DEF (posdef) / ERROR_SINGULAR
            nd%nelim = nd%ncol       ! keep downstream indexing sane
         END IF
         nd%ndelay_out = nd%ncol - nd%nelim
         IF (nd%parent == 0 .AND. nd%ndelay_out /= 0) node_ok = .FALSE.
         IF (ncc > 0) THEN
            DO c = 1, nnodes
               IF (node(c)%parent /= p) CYCLE
               ASSOCIATE (ch => node(c))
               IF (ch%symb_nrow - ch%symb_ncol > 0) THEN
                  CALL build_cache(ch, tls_pmap, cache)
                  DO k = 1, ch%symb_nrow-ch%symb_ncol
                     cache(k) = cache(k) - nd%ncol
                  END DO
                  CALL assemble_expected_contrib(1_ip_, ch%symb_nrow-ch%symb_ncol, &
                       ch%symb_nrow-ch%symb_ncol, cache, ch%contrib, &
                       ch%symb_nrow-ch%symb_ncol, nd%contrib, ncc)
               END IF
               END ASSOCIATE
            END DO
            IF (PRESENT(contribs) .AND. ALLOCATED(nd%contribs)) THEN
               DO ci = 1, SIZE(nd%contribs)
                  ASSOCIATE (ct => contribs(nd%contribs(ci)))
                  IF (ct%cn > 0) THEN
                     IF (ALLOCATED(cache)) DEALLOCATE(cache)
                     ALLOCATE(cache(ct%cn))
                     DO k = 1, ct%cn
                        cache(k) = tls_pmap(ct%rlist(k)) - nd%ncol
                     END DO
                     CALL assemble_expected_contrib(1_ip_, ct%cn, ct%cn, cache, &
                          ct%val, ct%cn, nd%contrib, ncc)
                  END IF
                  END ASSOCIATE
               END DO
            END IF
         END IF
      END ASSOCIATE
   END SUBROUTINE factor_one_node

   !> Factor the whole subtree with an OpenMP task DAG: one task per node, with
   !! dependencies encoding the elimination tree. A node's task writes its own
   !! sync slot and reads its parent's; since children are created before the
   !! parent (postorder) and read the parent's slot, the parent's write waits for
   !! all children (WAR) -- so a node runs as soon as its children are done, with
   !! no level barriers. Falls back to serial without OpenMP.
   SUBROUTINE factor_subtree_delay(node, nnodes, n, action, u, small, nb, &
                                   posdef, ok, contribs, small_subtree_threshold, &
                                   failed_tpp, alloc_ok)
      TYPE(dmf_node), INTENT(INOUT) :: node(:)
      INTEGER(ip_),  INTENT(IN)    :: nnodes, n, nb
      LOGICAL,        INTENT(IN)    :: action, posdef
      REAL(rp_),     INTENT(IN)    :: u, small
      LOGICAL,        INTENT(OUT)   :: ok
      TYPE(subtree_contrib_t), INTENT(IN), OPTIONAL :: contribs(:)
      INTEGER(long_), INTENT(IN), OPTIONAL :: small_subtree_threshold
      LOGICAL,        INTENT(IN), OPTIONAL :: failed_tpp
      LOGICAL,        INTENT(OUT), OPTIONAL :: alloc_ok   ! .false. on OOM
      LOGICAL :: ftpp, aok
      INTEGER(ip_) :: p, pp, q, lop
      INTEGER(ip_), ALLOCATABLE :: sync(:), lo(:)
      INTEGER(long_), ALLOCATABLE :: flops(:)
      LOGICAL, ALLOCATABLE :: is_root(:), skip_node(:)
      INTEGER(long_) :: thresh, own
      INTEGER(ip_) :: kk
      LOGICAL :: nok, naok

      ok = .TRUE.; aok = .TRUE.
      ftpp = .TRUE.; IF (PRESENT(failed_tpp)) ftpp = failed_tpp
      IF (nnodes <= 0) RETURN
      ALLOCATE(sync(0:nnodes)); sync = 0     ! slot 0 = sentinel for roots

      ! ---- small leaf subtrees: group a complete leaf subtree whose flop count
      ! is below small_subtree_threshold and factor it in a single (serial) task
      ! for cache locality (port of SmallLeaf{Symbolic,Numeric}Subtree). The
      ! group task keeps the exact dependencies of its root node, and its members
      ! form a complete subtree, so children are always factored before parents.
      ALLOCATE(is_root(nnodes), skip_node(nnodes))
      is_root = .FALSE.; skip_node = .FALSE.
      thresh = 0_long_
      IF (PRESENT(small_subtree_threshold)) thresh = small_subtree_threshold
      IF (thresh > 0_long_) THEN
         ALLOCATE(lo(nnodes), flops(0:nnodes))
         flops = 0_long_
         DO p = 1, nnodes
            lo(p) = p
         END DO
         DO p = 1, nnodes
            own = 0_long_
            DO kk = 0, node(p)%symb_ncol-1
               own = own + INT(node(p)%symb_nrow-kk, long_)**2
            END DO
            flops(p) = flops(p) + own
            ! penalise nodes that receive a foreign subtree contribution (a
            ! parttree boundary), exactly as the C++ (SymbolicNode::contrib),
            ! so a small-leaf group never crosses such a boundary
            IF (ALLOCATED(node(p)%contribs)) THEN
               IF (SIZE(node(p)%contribs) > 0) flops(p) = flops(p) + thresh
            END IF
            pp = node(p)%parent
            IF (pp >= 1) THEN
               flops(pp) = flops(pp) + flops(p)
               lo(pp) = MIN(lo(pp), lo(p))
            END IF
         END DO
         DO p = 1, nnodes
            pp = node(p)%parent
            IF (p > lo(p) .AND. flops(p) < thresh) THEN
               IF (pp == 0) THEN
                  is_root(p) = .TRUE.
               ELSE IF (flops(pp) >= thresh) THEN
                  is_root(p) = .TRUE.
               END IF
               IF (is_root(p)) THEN
                  DO q = lo(p), p-1
                     skip_node(q) = .TRUE.
                  END DO
               END IF
            END IF
         END DO
      END IF

      !$omp parallel default(shared) private(p, pp, q, lop, nok, naok)
      !$omp single
      DO p = 1, nnodes
         IF (skip_node(p)) CYCLE              ! folded into its group's root task
         pp = node(p)%parent
         IF (is_root(p)) THEN
            lop = lo(p)
            !$omp task firstprivate(p, pp, lop) private(q, nok, naok) default(shared) &
            !$omp      depend(inout: sync(p)) depend(in: sync(pp))
            DO q = lop, p                      ! factor the whole leaf subtree
               CALL factor_one_node(node, q, nnodes, n, action, u, small, nb,   &
                                    posdef, nok, contribs, failed_tpp = ftpp,   &
                                    node_aok = naok)
               IF (.NOT. nok) THEN
                  !$omp atomic write
                  ok = .FALSE.
               END IF
               IF (.NOT. naok) THEN
                  !$omp atomic write
                  aok = .FALSE.
               END IF
            END DO
            !$omp end task
         ELSE
            !$omp task firstprivate(p, pp) private(nok, naok) default(shared) &
            !$omp      depend(inout: sync(p)) depend(in: sync(pp))
            CALL factor_one_node(node, p, nnodes, n, action, u, small, nb,      &
                                 posdef, nok, contribs, failed_tpp = ftpp,      &
                                 node_aok = naok)
            IF (.NOT. nok) THEN
               !$omp atomic write
               ok = .FALSE.
            END IF
            IF (.NOT. naok) THEN
               !$omp atomic write
               aok = .FALSE.
            END IF
            !$omp end task
         END IF
      END DO
      !$omp end single
      ! the implicit barrier at end single guarantees every task has completed,
      ! so each team thread can now free its own threadprivate scratch (tls_pmap
      ! persists past the region otherwise and leaks one copy per worker thread)
      IF (ALLOCATED(tls_pmap)) DEALLOCATE(tls_pmap)
      !$omp end parallel

      IF (PRESENT(alloc_ok)) alloc_ok = aok
      DEALLOCATE(sync, is_root, skip_node)
      IF (ALLOCATED(lo)) DEALLOCATE(lo, flops)
   END SUBROUTINE factor_subtree_delay

   SUBROUTINE extract_contrib(nd, ct)
      TYPE(dmf_node),          INTENT(IN)  :: nd
      TYPE(subtree_contrib_t), INTENT(OUT) :: ct
      INTEGER(ip_) :: cm, i, j, k, lddelay, dc
      cm = nd%symb_nrow - nd%symb_ncol
      ct%cn = cm; ct%ndelay = nd%ndelay_out
      IF (cm > 0) THEN
         ALLOCATE(ct%rlist(cm)); ct%rlist = nd%rlist(nd%symb_ncol+1 : nd%symb_nrow)
         ALLOCATE(ct%val(cm,cm)); ct%val = nd%contrib
      END IF
      IF (ct%ndelay > 0) THEN
         ALLOCATE(ct%delay_perm(ct%ndelay))
         ct%delay_perm = nd%perm(nd%nelim+1 : nd%ncol)
         lddelay = ct%ndelay + cm
         ALLOCATE(ct%delay_val(lddelay, ct%ndelay)); ct%delay_val = 0._rp_
         DO i = 1, ct%ndelay
            dc = nd%nelim + i
            DO j = i, ct%ndelay
               ct%delay_val(j, i) = nd%lcol(nd%nelim + j, dc)
            END DO
            DO k = 1, cm
               ct%delay_val(ct%ndelay + k, i) = nd%lcol(nd%ncol + k, dc)
            END DO
         END DO
      END IF
   END SUBROUTINE extract_contrib

   SUBROUTINE build_cache(ch, pmap, cache)
      TYPE(dmf_node), INTENT(IN) :: ch
      INTEGER(ip_),  INTENT(IN) :: pmap(:)
      INTEGER(ip_), ALLOCATABLE, INTENT(INOUT) :: cache(:)
      INTEGER(ip_) :: kk, ncc
      ncc = ch%symb_nrow - ch%symb_ncol
      IF (ALLOCATED(cache)) DEALLOCATE(cache)
      ALLOCATE(cache(MAX(ncc,1_ip_)))
      DO kk = 1, ncc
         cache(kk) = pmap(ch%rlist(ch%symb_ncol + kk))
      END DO
   END SUBROUTINE build_cache

! ===================== tree solves (per phase, multi-RHS) =============
   SUBROUTINE subtree_solve_fwd_delay(node, nnodes, nrhs, x, ldx)
      TYPE(dmf_node), INTENT(IN)    :: node(:)
      INTEGER(ip_),  INTENT(IN)    :: nnodes, nrhs, ldx
      REAL(rp_),     INTENT(INOUT) :: x(ldx, *)
      INTEGER(ip_) :: p
      INTEGER(ip_), ALLOCATABLE :: gl(:)
      REAL(rp_),    ALLOCATABLE :: xf(:,:)
      DO p = 1, nnodes
         CALL gl_of(node(p), gl)
         ALLOCATE(xf(node(p)%nrow, nrhs))
         CALL gather(x, ldx, gl, node(p)%nrow, nrhs, xf)
         CALL ldlt_app_solve_fwd(node(p)%nrow, node(p)%nelim, node(p)%lcol, &
              node(p)%ldl, nrhs, xf, node(p)%nrow)
         CALL scatter(x, ldx, gl, node(p)%nrow, nrhs, xf)
         DEALLOCATE(xf, gl)
      END DO
   END SUBROUTINE subtree_solve_fwd_delay

   SUBROUTINE subtree_solve_diag_delay(node, nnodes, nrhs, x, ldx)
      TYPE(dmf_node), INTENT(IN)    :: node(:)
      INTEGER(ip_),  INTENT(IN)    :: nnodes, nrhs, ldx
      REAL(rp_),     INTENT(INOUT) :: x(ldx, *)
      INTEGER(ip_) :: p
      INTEGER(ip_), ALLOCATABLE :: gl(:)
      REAL(rp_),    ALLOCATABLE :: xe(:,:)
      DO p = 1, nnodes
         IF (node(p)%nelim <= 0) CYCLE
         ALLOCATE(gl(node(p)%nelim)); gl = node(p)%perm(1:node(p)%nelim)
         ALLOCATE(xe(node(p)%nelim, nrhs))
         CALL gather(x, ldx, gl, node(p)%nelim, nrhs, xe)
         CALL ldlt_app_solve_diag(node(p)%nelim, node(p)%d, nrhs, xe, node(p)%nelim)
         CALL scatter(x, ldx, gl, node(p)%nelim, nrhs, xe)
         DEALLOCATE(xe, gl)
      END DO
   END SUBROUTINE subtree_solve_diag_delay

   SUBROUTINE subtree_solve_bwd_delay(node, nnodes, nrhs, x, ldx)
      TYPE(dmf_node), INTENT(IN)    :: node(:)
      INTEGER(ip_),  INTENT(IN)    :: nnodes, nrhs, ldx
      REAL(rp_),     INTENT(INOUT) :: x(ldx, *)
      INTEGER(ip_) :: p
      INTEGER(ip_), ALLOCATABLE :: gl(:)
      REAL(rp_),    ALLOCATABLE :: xf(:,:)
      DO p = nnodes, 1, -1
         CALL gl_of(node(p), gl)
         ALLOCATE(xf(node(p)%nrow, nrhs))
         CALL gather(x, ldx, gl, node(p)%nrow, nrhs, xf)
         CALL ldlt_app_solve_bwd(node(p)%nrow, node(p)%nelim, node(p)%lcol, &
              node(p)%ldl, nrhs, xf, node(p)%nrow)
         CALL scatter(x, ldx, gl, node(p)%nrow, nrhs, xf)
         DEALLOCATE(xf, gl)
      END DO
   END SUBROUTINE subtree_solve_bwd_delay

   SUBROUTINE gl_of(nd, gl)
      TYPE(dmf_node), INTENT(IN) :: nd
      INTEGER(ip_), ALLOCATABLE, INTENT(INOUT) :: gl(:)
      INTEGER(ip_) :: i
      IF (ALLOCATED(gl)) DEALLOCATE(gl)
      ALLOCATE(gl(nd%nrow))
      DO i = 1, nd%ncol
         gl(i) = nd%perm(i)
      END DO
      DO i = nd%ncol+1, nd%nrow
         gl(i) = nd%rlist(i - nd%ndelay_in)
      END DO
   END SUBROUTINE gl_of

   SUBROUTINE gather(x, ldx, gl, m, nrhs, xf)
      INTEGER(ip_), INTENT(IN)  :: ldx, m, nrhs, gl(:)
      REAL(rp_),    INTENT(IN)  :: x(ldx, *)
      REAL(rp_),    INTENT(OUT) :: xf(m, nrhs)
      INTEGER(ip_) :: i, r
      DO r = 1, nrhs
         DO i = 1, m
            xf(i, r) = x(gl(i), r)
         END DO
      END DO
   END SUBROUTINE gather

   SUBROUTINE scatter(x, ldx, gl, m, nrhs, xf)
      INTEGER(ip_), INTENT(IN)    :: ldx, m, nrhs, gl(:)
      REAL(rp_),    INTENT(INOUT) :: x(ldx, *)
      REAL(rp_),    INTENT(IN)    :: xf(m, nrhs)
      INTEGER(ip_) :: i, r
      DO r = 1, nrhs
         DO i = 1, m
            x(gl(i), r) = xf(i, r)
         END DO
      END DO
   END SUBROUTINE scatter

 END MODULE GALAHAD_SSIDS_factor_precision
