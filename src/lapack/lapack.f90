! THIS VERSION: GALAHAD 5.1 - 2024-11-30 AT 16:00 GMT

#include "galahad_lapack.h"

! Reference lapack, see http://www.netlib.org/lapack/explore-html/

! C preprocessor will change, as appropriate,
!  ip_ to INT32 or INT64, 
!  rp_ to REAL32, REAL64 or REAL128
!  subroutine and function names starting with D to start with S or Q

! for compilers that do not support IEEE arthmetic, use a -DNO_IEECK flag

        SUBROUTINE DBDSQR(uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu,  &
          c, ldc, work, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, ldc, ldu, ldvt, n, ncc, ncvt, nru
          REAL(rp_) :: c(ldc, *), d(*), e(*), u(ldu, *), vt(ldvt, *),       &
            work(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          REAL(rp_) :: negone
          PARAMETER (negone=-1.0_rp_)
          REAL(rp_) :: hndrth
          PARAMETER (hndrth=0.01_rp_)
          REAL(rp_) :: ten
          PARAMETER (ten=10.0_rp_)
          REAL(rp_) :: hndrd
          PARAMETER (hndrd=100.0_rp_)
          REAL(rp_) :: meigth
          PARAMETER (meigth=-0.125_rp_)
          INTEGER(ip_) :: maxitr
          PARAMETER (maxitr=6)
          LOGICAL :: lower, rotate
          INTEGER(ip_) :: i, idir, isub, iter, iterdivn, j, ll, lll, m,     &
            maxitdivn, nm1, nm12, nm13, oldll, oldm
          REAL(rp_) :: abse, abss, cosl, cosr, cs, eps, f, g, h, mu, oldcs, &
            oldsn, r, shift, sigmn, sigmx, sinl, sinr, sll, smax, smin,     &
            sminl, sminoa, sn, thresh, tol, tolmul, unfl, rn
          LOGICAL :: LSAME
          REAL(rp_) :: DLAMCH
          EXTERNAL :: LSAME, DLAMCH
          EXTERNAL :: DLARTG, DLAS2, DLASQ1, DLASR, DLASV2,& 
            DROT, DSCAL, DSWAP, XERBLA2
          INTRINSIC :: ABS, REAL, MAX, MIN, SIGN, SQRT
          info = 0
          lower = LSAME(uplo, 'L')
          IF (.NOT. LSAME(uplo,'U') .AND. .NOT. lower) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (ncvt<0) THEN
            info = -3
          ELSE IF (nru<0) THEN
            info = -4
          ELSE IF (ncc<0) THEN
            info = -5
          ELSE IF ((ncvt==0 .AND. ldvt<1) .OR. (ncvt>0 .AND. ldvt<MAX(1,    &
            n))) THEN
            info = -9
          ELSE IF (ldu<MAX(1,nru)) THEN
            info = -11
          ELSE IF ((ncc==0 .AND. ldc<1) .OR. (ncc>0 .AND. ldc<MAX(1, n)))   &
            THEN
            info = -13
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('BDSQR', -info)
            RETURN
          END IF
          IF (n==0) RETURN
          IF (n==1) GO TO 160
          rotate = (ncvt>0) .OR. (nru>0) .OR. (ncc>0)
          IF (.NOT. rotate) THEN
            CALL DLASQ1(n, d, e, work, info)
            IF (info/=2) RETURN
            info = 0
          END IF
          nm1 = n - 1
          nm12 = nm1 + nm1
          nm13 = nm12 + nm1
          idir = 0
          eps = DLAMCH('Epsilon')
          unfl = DLAMCH('Safe minimum')
          IF (lower) THEN
            DO i = 1, n - 1
              CALL DLARTG(d(i), e(i), cs, sn, r)
              d(i) = r
              e(i) = sn*d(i+1)
              d(i+1) = cs*d(i+1)
              work(i) = cs
              work(nm1+i) = sn
            END DO
            IF (nru>0) CALL DLASR('R', 'V', 'F', nru, n, work(1), work(n),  &
              u, ldu)
            IF (ncc>0) CALL DLASR('L', 'V', 'F', n, ncc, work(1), work(n),  &
              c, ldc)
          END IF
          tolmul = MAX(ten, MIN(hndrd,eps**meigth))
          tol = tolmul*eps
          smax = zero
          DO i = 1, n
            smax = MAX(smax, ABS(d(i)))
          END DO
          DO i = 1, n - 1
            smax = MAX(smax, ABS(e(i)))
          END DO
          sminl = zero
          IF (tol>=zero) THEN
            sminoa = ABS(d(1))
            IF (sminoa==zero) GO TO 50
            mu = sminoa
            DO i = 2, n
              mu = ABS(d(i))*(mu/(mu+ABS(e(i-1))))
              sminoa = MIN(sminoa, mu)
              IF (sminoa==zero) GO TO 50
            END DO
 50         CONTINUE
            sminoa = sminoa/SQRT(REAL(n,rp_))
            rn = REAL(n,rp_)
            thresh = MAX(tol*sminoa, REAL(maxitr,rp_)*(rn*(rn*unfl)))
          ELSE
            rn = REAL(n,rp_)
            thresh = MAX(ABS(tol)*smax, REAL(maxitr,rp_)*(rn*(rn*unfl)))
          END IF
          maxitdivn = maxitr*n
          iterdivn = 0
          iter = -1
          oldll = -1
          oldm = -1
          m = n
 60       CONTINUE
          IF (m<=1) GO TO 160
          IF (iter>=n) THEN
            iter = iter - n
            iterdivn = iterdivn + 1
            IF (iterdivn>=maxitdivn) GO TO 200
          END IF
          IF (tol<zero .AND. ABS(d(m))<=thresh) d(m) = zero
          smax = ABS(d(m))
          smin = smax
          DO lll = 1, m - 1
            ll = m - lll
            abss = ABS(d(ll))
            abse = ABS(e(ll))
            IF (tol<zero .AND. abss<=thresh) d(ll) = zero
            IF (abse<=thresh) GO TO 80
            smin = MIN(smin, abss)
            smax = MAX(smax, abss, abse)
          END DO
          ll = 0
          GO TO 90
 80       CONTINUE
          e(ll) = zero
          IF (ll==m-1) THEN
            m = m - 1
            GO TO 60
          END IF
 90       CONTINUE
          ll = ll + 1
          IF (ll==m-1) THEN
            CALL DLASV2(d(m-1), e(m-1), d(m), sigmn, sigmx, sinr, cosr,     &
              sinl, cosl)
            d(m-1) = sigmx
            e(m-1) = zero
            d(m) = sigmn
            IF (ncvt>0) CALL DROT(ncvt, vt(m-1,1), ldvt, vt(m,1), ldvt,     &
              cosr, sinr)
            IF (nru>0) CALL DROT(nru, u(1,m-1), 1_ip_, u(1,m), 1_ip_, cosl, &
              sinl)
            IF (ncc>0) CALL DROT(ncc, c(m-1,1), ldc, c(m,1), ldc, cosl,     &
              sinl)
            m = m - 2
            GO TO 60
          END IF
          IF (ll>oldm .OR. m<oldll) THEN
            IF (ABS(d(ll))>=ABS(d(m))) THEN
              idir = 1
            ELSE
              idir = 2
            END IF
          END IF
          IF (idir==1) THEN
            IF (ABS(e(m-1))<=ABS(tol)*ABS(d(m)) .OR. (tol<zero .AND. ABS(e  &
              (m-1))<=thresh)) THEN
              e(m-1) = zero
              GO TO 60
            END IF
            IF (tol>=zero) THEN
              mu = ABS(d(ll))
              sminl = mu
              DO lll = ll, m - 1
                IF (ABS(e(lll))<=tol*mu) THEN
                  e(lll) = zero
                  GO TO 60
                END IF
                mu = ABS(d(lll+1))*(mu/(mu+ABS(e(lll))))
                sminl = MIN(sminl, mu)
              END DO
            END IF
          ELSE
            IF (ABS(e(ll))<=ABS(tol)*ABS(d(ll)) .OR. (tol<zero .AND. ABS(   &
              e(ll))<=thresh)) THEN
              e(ll) = zero
              GO TO 60
            END IF
            IF (tol>=zero) THEN
              mu = ABS(d(m))
              sminl = mu
              DO lll = m - 1, ll, -1_ip_
                IF (ABS(e(lll))<=tol*mu) THEN
                  e(lll) = zero
                  GO TO 60
                END IF
                mu = ABS(d(lll))*(mu/(mu+ABS(e(lll))))
                sminl = MIN(sminl, mu)
              END DO
            END IF
          END IF
          oldll = ll
          oldm = m
          IF (tol>=zero .AND.                                               &
              REAL(n,rp_)*tol*(sminl/smax)<=MAX(eps,hndrth*tol)) THEN
            shift = zero
          ELSE
            IF (idir==1) THEN
              sll = ABS(d(ll))
              CALL DLAS2(d(m-1), e(m-1), d(m), shift, r)
            ELSE
              sll = ABS(d(m))
              CALL DLAS2(d(ll), e(ll), d(ll+1), shift, r)
            END IF
            IF (sll>zero) THEN
              IF ((shift/sll)**2<eps) shift = zero
            END IF
          END IF
          iter = iter + m - ll
          IF (shift==zero) THEN
            IF (idir==1) THEN
              cs = one
              oldcs = one
              DO i = ll, m - 1
                CALL DLARTG(d(i)*cs, e(i), cs, sn, r)
                IF (i>ll) e(i-1) = oldsn*r
                CALL DLARTG(oldcs*r, d(i+1)*sn, oldcs, oldsn, d(i))
                work(i-ll+1) = cs
                work(i-ll+1+nm1) = sn
                work(i-ll+1+nm12) = oldcs
                work(i-ll+1+nm13) = oldsn
              END DO
              h = d(m)*cs
              d(m) = h*oldcs
              e(m-1) = h*oldsn
              IF (ncvt>0) CALL DLASR('L', 'V', 'F', m-ll+1, ncvt, work(1),  &
                work(n), vt(ll,1), ldvt)
              IF (nru>0) CALL DLASR('R', 'V', 'F', nru, m-ll+1,             &
                work(nm12+1), work(nm13+1), u(1,ll), ldu)
              IF (ncc>0) CALL DLASR('L', 'V', 'F', m-ll+1, ncc,             &
                work(nm12+1), work(nm13+1), c(ll,1), ldc)
              IF (ABS(e(m-1))<=thresh) e(m-1) = zero
            ELSE
              cs = one
              oldcs = one
              DO i = m, ll + 1, -1_ip_
                CALL DLARTG(d(i)*cs, e(i-1), cs, sn, r)
                IF (i<m) e(i) = oldsn*r
                CALL DLARTG(oldcs*r, d(i-1)*sn, oldcs, oldsn, d(i))
                work(i-ll) = cs
                work(i-ll+nm1) = -sn
                work(i-ll+nm12) = oldcs
                work(i-ll+nm13) = -oldsn
              END DO
              h = d(ll)*cs
              d(ll) = h*oldcs
              e(ll) = h*oldsn
              IF (ncvt>0) CALL DLASR('L', 'V', 'B', m-ll+1, ncvt,           &
                work(nm12+1), work(nm13+1), vt(ll,1), ldvt)
              IF (nru>0) CALL DLASR('R', 'V', 'B', nru, m-ll+1, work(1),    &
                work(n), u(1,ll), ldu)
              IF (ncc>0) CALL DLASR('L', 'V', 'B', m-ll+1, ncc, work(1),    &
                work(n), c(ll,1), ldc)
              IF (ABS(e(ll))<=thresh) e(ll) = zero
            END IF
          ELSE
            IF (idir==1) THEN
              f = (ABS(d(ll))-shift)*(SIGN(one,d(ll))+shift/d(ll))
              g = e(ll)
              DO i = ll, m - 1
                CALL DLARTG(f, g, cosr, sinr, r)
                IF (i>ll) e(i-1) = r
                f = cosr*d(i) + sinr*e(i)
                e(i) = cosr*e(i) - sinr*d(i)
                g = sinr*d(i+1)
                d(i+1) = cosr*d(i+1)
                CALL DLARTG(f, g, cosl, sinl, r)
                d(i) = r
                f = cosl*e(i) + sinl*d(i+1)
                d(i+1) = cosl*d(i+1) - sinl*e(i)
                IF (i<m-1) THEN
                  g = sinl*e(i+1)
                  e(i+1) = cosl*e(i+1)
                END IF
                work(i-ll+1) = cosr
                work(i-ll+1+nm1) = sinr
                work(i-ll+1+nm12) = cosl
                work(i-ll+1+nm13) = sinl
              END DO
              e(m-1) = f
              IF (ncvt>0) CALL DLASR('L', 'V', 'F', m-ll+1, ncvt, work(1),  &
                work(n), vt(ll,1), ldvt)
              IF (nru>0) CALL DLASR('R', 'V', 'F', nru, m-ll+1,             &
                work(nm12+1), work(nm13+1), u(1,ll), ldu)
              IF (ncc>0) CALL DLASR('L', 'V', 'F', m-ll+1, ncc,             &
                work(nm12+1), work(nm13+1), c(ll,1), ldc)
              IF (ABS(e(m-1))<=thresh) e(m-1) = zero
            ELSE
              f = (ABS(d(m))-shift)*(SIGN(one,d(m))+shift/d(m))
              g = e(m-1)
              DO i = m, ll + 1, -1_ip_
                CALL DLARTG(f, g, cosr, sinr, r)
                IF (i<m) e(i) = r
                f = cosr*d(i) + sinr*e(i-1)
                e(i-1) = cosr*e(i-1) - sinr*d(i)
                g = sinr*d(i-1)
                d(i-1) = cosr*d(i-1)
                CALL DLARTG(f, g, cosl, sinl, r)
                d(i) = r
                f = cosl*e(i-1) + sinl*d(i-1)
                d(i-1) = cosl*d(i-1) - sinl*e(i-1)
                IF (i>ll+1) THEN
                  g = sinl*e(i-2)
                  e(i-2) = cosl*e(i-2)
                END IF
                work(i-ll) = cosr
                work(i-ll+nm1) = -sinr
                work(i-ll+nm12) = cosl
                work(i-ll+nm13) = -sinl
              END DO
              e(ll) = f
              IF (ABS(e(ll))<=thresh) e(ll) = zero
              IF (ncvt>0) CALL DLASR('L', 'V', 'B', m-ll+1, ncvt,           &
                work(nm12+1), work(nm13+1), vt(ll,1), ldvt)
              IF (nru>0) CALL DLASR('R', 'V', 'B', nru, m-ll+1, work(1),    &
                work(n), u(1,ll), ldu)
              IF (ncc>0) CALL DLASR('L', 'V', 'B', m-ll+1, ncc, work(1),    &
                work(n), c(ll,1), ldc)
            END IF
          END IF
          GO TO 60
 160      CONTINUE
          DO i = 1, n
            IF (d(i)<zero) THEN
              d(i) = -d(i)
              IF (ncvt>0) CALL DSCAL(ncvt, negone, vt(i,1), ldvt)
            END IF
          END DO
          DO i = 1, n - 1
            isub = 1
            smin = d(1)
            DO j = 2, n + 1 - i
              IF (d(j)<=smin) THEN
                isub = j
                smin = d(j)
              END IF
            END DO
            IF (isub/=n+1-i) THEN
              d(isub) = d(n+1-i)
              d(n+1-i) = smin
              IF (ncvt>0) CALL DSWAP(ncvt, vt(isub,1), ldvt, vt(n+1-i,1),   &
                ldvt)
              IF (nru>0) CALL DSWAP(nru, u(1,isub), 1_ip_, u(1,n+1-i),      &
                1_ip_)
              IF (ncc>0) CALL DSWAP(ncc, c(isub,1), ldc, c(n+1-i,1), ldc)
            END IF
          END DO
          GO TO 220
 200      CONTINUE
          info = 0
          DO i = 1, n - 1
            IF (e(i)/=zero) info = info + 1
          END DO
 220      CONTINUE
          RETURN
        END SUBROUTINE

        SUBROUTINE DCOMBSSQ(v1, v2)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: v1(2), v2(2)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          IF (v1(1)>=v2(1)) THEN
            IF (v1(1)/=zero) THEN
              v1(2) = v1(2) + (v2(1)/v1(1))**2*v2(2)
            ELSE
              v1(2) = v1(2) + v2(2)
            END IF
          ELSE
            v1(2) = v2(2) + (v1(1)/v2(1))**2*v1(2)
            v1(1) = v2(1)
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DGEBD2(m, n, a, lda, d, e, tauq, taup, work, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, lda, m, n
          REAL(rp_) :: a(lda, *), d(*), e(*), taup(*), tauq(*), work(*)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          INTEGER(ip_) :: i
          EXTERNAL :: DLARF, DLARFG, XERBLA2
          INTRINSIC :: MAX, MIN
          info = 0
          IF (m<0) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,m)) THEN
            info = -4
          END IF
          IF (info<0) THEN
            CALL XERBLA2('GEBD2', -info)
            RETURN
          END IF
          IF (m>=n) THEN
            DO i = 1, n
              CALL DLARFG(m-i+1, a(i,i), a(MIN(i+1,m),i), 1_ip_, tauq(i))
              d(i) = a(i, i)
              a(i, i) = one
              IF (i<n) CALL DLARF('Left', m-i+1, n-i, a(i,i), 1_ip_,        &
                tauq(i), a(i,i+1), lda, work)
              a(i, i) = d(i)
              IF (i<n) THEN
                CALL DLARFG(n-i, a(i,i+1), a(i,MIN(i+2,n)), lda, taup(i))
                e(i) = a(i, i+1)
                a(i, i+1) = one
                CALL DLARF('Right', m-i, n-i, a(i,i+1), lda, taup(i),       &
                  a(i+1,i+1), lda, work)
                a(i, i+1) = e(i)
              ELSE
                taup(i) = zero
              END IF
            END DO
          ELSE
            DO i = 1, m
              CALL DLARFG(n-i+1, a(i,i), a(i,MIN(i+1,n)), lda, taup(i))
              d(i) = a(i, i)
              a(i, i) = one
              IF (i<m) CALL DLARF('Right', m-i, n-i+1, a(i,i), lda,         &
                taup(i), a(i+1,i), lda, work)
              a(i, i) = d(i)
              IF (i<m) THEN
                CALL DLARFG(m-i, a(i+1,i), a(MIN(i+2,m),i), 1_ip_, tauq(i))
                e(i) = a(i+1, i)
                a(i+1, i) = one
                CALL DLARF('Left', m-i, n-i, a(i+1,i), 1_ip_, tauq(i),      &
                  a(i+1,i+1), lda, work)
                a(i+1, i) = e(i)
              ELSE
                tauq(i) = zero
              END IF
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DGEBRD(m, n, a, lda, d, e, tauq, taup, work, lwork,      &
          info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, lda, lwork, m, n
          REAL(rp_) :: a(lda, *), d(*), e(*), taup(*), tauq(*), work(*)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          LOGICAL :: lquery
          INTEGER(ip_) :: i, iinfo, j, ldwrkx, ldwrky, lwkopt, minmn, nb,   &
            nbmin, nx, ws
          EXTERNAL :: DGEBD2, DGEMM, DLABRD, XERBLA2
          INTRINSIC :: REAL, MAX, MIN
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: ILAENV2
          info = 0
          nb = MAX(1_ip_, ILAENV2(1_ip_,'GEBRD',' ',m,n,-1_ip_,-1_ip_))
          lwkopt = (m+n)*nb
          work(1) = REAL(lwkopt,rp_)
          lquery = (lwork==-1)
          IF (m<0) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,m)) THEN
            info = -4
          ELSE IF (lwork<MAX(1,m,n) .AND. .NOT. lquery) THEN
            info = -10
          END IF
          IF (info<0) THEN
            CALL XERBLA2('GEBRD', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          minmn = MIN(m, n)
          IF (minmn==0) THEN
            work(1) = 1
            RETURN
          END IF
          ws = MAX(m, n)
          ldwrkx = m
          ldwrky = n
          IF (nb>1 .AND. nb<minmn) THEN
            nx = MAX(nb, ILAENV2(3_ip_,'GEBRD',' ',m,n,-1_ip_,-1_ip_))
            IF (nx<minmn) THEN
              ws = (m+n)*nb
              IF (lwork<ws) THEN
                nbmin = ILAENV2(2_ip_, 'GEBRD', ' ', m, n, -1_ip_, -1_ip_)
                IF (lwork>=(m+n)*nbmin) THEN
                  nb = lwork/(m+n)
                ELSE
                  nb = 1
                  nx = minmn
                END IF
              END IF
            END IF
          ELSE
            nx = minmn
          END IF
          DO i = 1, minmn - nx, nb
            CALL DLABRD(m-i+1, n-i+1, nb, a(i,i), lda, d(i), e(i), tauq(i), &
              taup(i), work, ldwrkx, work(ldwrkx*nb+1), ldwrky)
            CALL DGEMM('No transpose', 'Transpose', m-i-nb+1, n-i-nb+1, nb, &
              -one, a(i+nb,i), lda, work(ldwrkx*nb+nb+1), ldwrky, one,      &
              a(i+nb,i+nb), lda)
            CALL DGEMM('No transpose', 'No transpose', m-i-nb+1, n-i-nb+1,  &
              nb, -one, work(nb+1), ldwrkx, a(i,i+nb), lda, one, a(i+nb,    &
              i+nb), lda)
            IF (m>=n) THEN
              DO j = i, i + nb - 1
                a(j, j) = d(j)
                a(j, j+1) = e(j)
              END DO
            ELSE
              DO j = i, i + nb - 1
                a(j, j) = d(j)
                a(j+1, j) = e(j)
              END DO
            END IF
          END DO
          CALL DGEBD2(m-i+1, n-i+1, a(i,i), lda, d(i), e(i), tauq(i),       &
            taup(i), work, iinfo)
          work(1) = REAL(ws,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DGEHD2(n, ilo, ihi, a, lda, tau, work, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: ihi, ilo, info, lda, n
          REAL(rp_) :: a(lda, *), tau(*), work(*)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          INTEGER(ip_) :: i
          REAL(rp_) :: aii
          EXTERNAL :: DLARF, DLARFG, XERBLA2
          INTRINSIC :: MAX, MIN
          info = 0
          IF (n<0) THEN
            info = -1
          ELSE IF (ilo<1 .OR. ilo>MAX(1,n)) THEN
            info = -2
          ELSE IF (ihi<MIN(ilo,n) .OR. ihi>n) THEN
            info = -3
          ELSE IF (lda<MAX(1,n)) THEN
            info = -5
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GEHD2', -info)
            RETURN
          END IF
          DO i = ilo, ihi - 1
            CALL DLARFG(ihi-i, a(i+1,i), a(MIN(i+2,n),i), 1_ip_, tau(i))
            aii = a(i+1, i)
            a(i+1, i) = one
            CALL DLARF('Right', ihi, ihi-i, a(i+1,i), 1_ip_, tau(i), a(1,   &
              i+1), lda, work)
            CALL DLARF('Left', ihi-i, n-i, a(i+1,i), 1_ip_, tau(i), a(i+1,  &
              i+1), lda, work)
            a(i+1, i) = aii
          END DO
          RETURN
        END SUBROUTINE

        SUBROUTINE DGEHRD(n, ilo, ihi, a, lda, tau, work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: ihi, ilo, info, lda, lwork, n
          REAL(rp_) :: a(lda, *), tau(*), work(*)
          INTEGER(ip_) :: nbmax, ldt, tsize
          PARAMETER (nbmax=64, ldt=nbmax+1, tsize=ldt*nbmax)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          LOGICAL :: lquery
          INTEGER(ip_) :: i, ib, iinfo, iwt, j, ldwork, lwkopt, nb, nbmin,  &
            nh, nx
          REAL(rp_) :: ei
          EXTERNAL :: DAXPY, DGEHD2, DGEMM, DLAHR2, DLARFB,& 
            DTRMM, XERBLA2
          INTRINSIC :: MAX, MIN
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: ILAENV2
          info = 0
          lquery = (lwork==-1)
          IF (n<0) THEN
            info = -1
          ELSE IF (ilo<1 .OR. ilo>MAX(1,n)) THEN
            info = -2
          ELSE IF (ihi<MIN(ilo,n) .OR. ihi>n) THEN
            info = -3
          ELSE IF (lda<MAX(1,n)) THEN
            info = -5
          ELSE IF (lwork<MAX(1,n) .AND. .NOT. lquery) THEN
            info = -8
          END IF
          IF (info==0) THEN
            nb = MIN(nbmax, ILAENV2(1_ip_,'GEHRD',' ',n,ilo,ihi,-1_ip_))
            lwkopt = n*nb + tsize
            work(1) = REAL(lwkopt,rp_)
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GEHRD', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          DO i = 1, ilo - 1
            tau(i) = zero
          END DO
          DO i = MAX(1, ihi), n - 1
            tau(i) = zero
          END DO
          nh = ihi - ilo + 1
          IF (nh<=1) THEN
            work(1) = 1
            RETURN
          END IF
          nb = MIN(nbmax, ILAENV2(1_ip_,'GEHRD',' ',n,ilo,ihi,-1_ip_))
          nbmin = 2
          IF (nb>1 .AND. nb<nh) THEN
            nx = MAX(nb, ILAENV2(3_ip_,'GEHRD',' ',n,ilo,ihi,-1_ip_))
            IF (nx<nh) THEN
              IF (lwork<n*nb+tsize) THEN
                nbmin = MAX(2_ip_, ILAENV2(2_ip_,'GEHRD',' ',n,ilo,ihi,     &
                  -1_ip_))
                IF (lwork>=(n*nbmin+tsize)) THEN
                  nb = (lwork-tsize)/n
                ELSE
                  nb = 1
                END IF
              END IF
            END IF
          END IF
          ldwork = n
          IF (nb<nbmin .OR. nb>=nh) THEN
            i = ilo
          ELSE
            iwt = 1 + n*nb
            DO i = ilo, ihi - 1 - nx, nb
              ib = MIN(nb, ihi-i)
              CALL DLAHR2(ihi, i, ib, a(1,i), lda, tau(i), work(iwt), ldt,  &
                work, ldwork)
              ei = a(i+ib, i+ib-1)
              a(i+ib, i+ib-1) = one
              CALL DGEMM('No transpose', 'Transpose', ihi, ihi-i-ib+1, ib,  &
                -one, work, ldwork, a(i+ib,i), lda, one, a(1,i+ib), lda)
              a(i+ib, i+ib-1) = ei
              CALL DTRMM('Right', 'Lower', 'Transpose', 'Unit', i, ib-1,    &
                one, a(i+1,i), lda, work, ldwork)
              DO j = 0, ib - 2
                CALL DAXPY(i, -one, work(ldwork*j+1), 1_ip_, a(1,i+j+1),    &
                  1_ip_)
              END DO
              CALL DLARFB('Left', 'Transpose', 'Forward', 'Columnwise',     &
                ihi-i, n-i-ib+1, ib, a(i+1,i), lda, work(iwt), ldt, a(i+1,  &
                i+ib), lda, work, ldwork)
            END DO
          END IF
          CALL DGEHD2(n, i, ihi, a, lda, tau, work, iinfo)
          work(1) = REAL(lwkopt,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DGELQ2(m, n, a, lda, tau, work, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, lda, m, n
          REAL(rp_) :: a(lda, *), tau(*), work(*)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          INTEGER(ip_) :: i, k
          REAL(rp_) :: aii
          EXTERNAL :: DLARF, DLARFG, XERBLA2
          INTRINSIC :: MAX, MIN
          info = 0
          IF (m<0) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,m)) THEN
            info = -4
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GELQ2', -info)
            RETURN
          END IF
          k = MIN(m, n)
          DO i = 1, k
            CALL DLARFG(n-i+1, a(i,i), a(i,MIN(i+1,n)), lda, tau(i))
            IF (i<m) THEN
              aii = a(i, i)
              a(i, i) = one
              CALL DLARF('Right', m-i, n-i+1, a(i,i), lda, tau(i), a(i+1,   &
                i), lda, work)
              a(i, i) = aii
            END IF
          END DO
          RETURN
        END SUBROUTINE

        SUBROUTINE DGELQF(m, n, a, lda, tau, work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, lda, lwork, m, n
          REAL(rp_) :: a(lda, *), tau(*), work(*)
          LOGICAL :: lquery
          INTEGER(ip_) :: i, ib, iinfo, iws, k, ldwork, lwkopt, nb, nbmin,  &
            nx
          EXTERNAL :: DGELQ2, DLARFB, DLARFT, XERBLA2
          INTRINSIC :: MAX, MIN
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: ILAENV2
          info = 0
          nb = ILAENV2(1_ip_, 'GELQF', ' ', m, n, -1_ip_, -1_ip_)
          lwkopt = m*nb
          work(1) = REAL(lwkopt,rp_)
          lquery = (lwork==-1)
          IF (m<0) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,m)) THEN
            info = -4
          ELSE IF (lwork<MAX(1,m) .AND. .NOT. lquery) THEN
            info = -7
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GELQF', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          k = MIN(m, n)
          IF (k==0) THEN
            work(1) = 1
            RETURN
          END IF
          nbmin = 2
          nx = 0
          iws = m
          IF (nb>1 .AND. nb<k) THEN
            nx = MAX(0_ip_, ILAENV2(3_ip_,'GELQF',' ',m,n,-1_ip_,-1_ip_))
            IF (nx<k) THEN
              ldwork = m
              iws = ldwork*nb
              IF (lwork<iws) THEN
                nb = lwork/ldwork
                nbmin = MAX(2_ip_, ILAENV2(2_ip_,'GELQF',' ',m,n,-1_ip_,    &
                  -1_ip_))
              END IF
            END IF
          END IF
          IF (nb>=nbmin .AND. nb<k .AND. nx<k) THEN
            DO i = 1, k - nx, nb
              ib = MIN(k-i+1, nb)
              CALL DGELQ2(ib, n-i+1, a(i,i), lda, tau(i), work, iinfo)
              IF (i+ib<=m) THEN
                CALL DLARFT('Forward', 'Rowwise', n-i+1, ib, a(i,i), lda,   &
                  tau(i), work, ldwork)
                CALL DLARFB('Right', 'No transpose', 'Forward', 'Rowwise',  &
                  m-i-ib+1, n-i+1, ib, a(i,i), lda, work, ldwork, a(i+ib,i),&
                  lda, work(ib+1), ldwork)
              END IF
            END DO
          ELSE
            i = 1
          END IF
          IF (i<=k) CALL DGELQ2(m-i+1, n-i+1, a(i,i), lda, tau(i), work,    &
            iinfo)
          work(1) = REAL(iws,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DGELS(trans, m, n, nrhs, a, lda, b, ldb, work, lwork,    &
          info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: trans
          INTEGER(ip_) :: info, lda, ldb, lwork, m, n, nrhs
          REAL(rp_) :: a(lda, *), b(ldb, *), work(*)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          LOGICAL :: lquery, tpsd
          INTEGER(ip_) :: brow, i, iascl, ibscl, j, mn, nb, scllen, wsize
          REAL(rp_) :: anrm, bignum, bnrm, smlnum
          REAL(rp_) :: rwork(1)
          LOGICAL :: LSAME
          INTEGER(ip_) :: ILAENV2
          REAL(rp_) :: DLAMCH, DLANGE
          EXTERNAL :: LSAME, ILAENV2, DLABAD, DLAMCH, DLANGE
          EXTERNAL :: DGELQF, DGEQRF, DLASCL, DLASET, &
            DORMLQ, DORMQR, DTRTRS, XERBLA2
          INTRINSIC :: REAL, MAX, MIN
          info = 0
          mn = MIN(m, n)
          lquery = (lwork==-1)
          IF (.NOT. (LSAME(trans,'N') .OR. LSAME(trans,'T'))) THEN
            info = -1
          ELSE IF (m<0) THEN
            info = -2
          ELSE IF (n<0) THEN
            info = -3
          ELSE IF (nrhs<0) THEN
            info = -4
          ELSE IF (lda<MAX(1,m)) THEN
            info = -6
          ELSE IF (ldb<MAX(1,m,n)) THEN
            info = -8
          ELSE IF (lwork<MAX(1,mn+MAX(mn,nrhs)) .AND. .NOT. lquery) THEN
            info = -10
          END IF
          IF (info==0 .OR. info==-10) THEN
            tpsd = .TRUE.
            IF (LSAME(trans,'N')) tpsd = .FALSE.
            IF (m>=n) THEN
              nb = ILAENV2(1_ip_, 'GEQRF', ' ', m, n, -1_ip_, -1_ip_)
              IF (tpsd) THEN
                nb = MAX(nb, ILAENV2(1_ip_,'ORMQR','LN',m,nrhs,n,-1_ip_))
              ELSE
                nb = MAX(nb, ILAENV2(1_ip_,'ORMQR','LT',m,nrhs,n,-1_ip_))
              END IF
            ELSE
              nb = ILAENV2(1_ip_, 'GELQF', ' ', m, n, -1_ip_, -1_ip_)
              IF (tpsd) THEN
                nb = MAX(nb, ILAENV2(1_ip_,'ORMLQ','LT',n,nrhs,m,-1_ip_))
              ELSE
                nb = MAX(nb, ILAENV2(1_ip_,'ORMLQ','LN',n,nrhs,m,-1_ip_))
              END IF
            END IF
            wsize = MAX(1, mn+MAX(mn,nrhs)*nb)
            work(1) = REAL(wsize,rp_)
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GELS ', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          IF (MIN(m,n,nrhs)==0) THEN
            CALL DLASET('Full', MAX(m,n), nrhs, zero, zero, b, ldb)
            RETURN
          END IF
          smlnum = DLAMCH('S')/DLAMCH('P')
          bignum = one/smlnum
          CALL DLABAD(smlnum, bignum)
          anrm = DLANGE('M', m, n, a, lda, rwork)
          iascl = 0
          IF (anrm>zero .AND. anrm<smlnum) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, smlnum, m, n, a, lda,      &
              info)
            iascl = 1
          ELSE IF (anrm>bignum) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, bignum, m, n, a, lda,      &
              info)
            iascl = 2
          ELSE IF (anrm==zero) THEN
            CALL DLASET('F', MAX(m,n), nrhs, zero, zero, b, ldb)
            GO TO 50
          END IF
          brow = m
          IF (tpsd) brow = n
          bnrm = DLANGE('M', brow, nrhs, b, ldb, rwork)
          ibscl = 0
          IF (bnrm>zero .AND. bnrm<smlnum) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, bnrm, smlnum, brow, nrhs, b,     &
              ldb, info)
            ibscl = 1
          ELSE IF (bnrm>bignum) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, bnrm, bignum, brow, nrhs, b,     &
              ldb, info)
            ibscl = 2
          END IF
          IF (m>=n) THEN
            CALL DGEQRF(m, n, a, lda, work(1), work(mn+1), lwork-mn, info)
            IF (.NOT. tpsd) THEN
              CALL DORMQR('Left', 'Transpose', m, nrhs, n, a, lda, work(1), &
                b, ldb, work(mn+1), lwork-mn, info)
              CALL DTRTRS('Upper', 'No transpose', 'Non-unit', n, nrhs, a,  &
                lda, b, ldb, info)
              IF (info>0) THEN
                RETURN
              END IF
              scllen = n
            ELSE
              CALL DTRTRS('Upper', 'Transpose', 'Non-unit', n, nrhs, a,     &
                lda, b, ldb, info)
              IF (info>0) THEN
                RETURN
              END IF
              DO j = 1, nrhs
                DO i = n + 1, m
                  b(i, j) = zero
                END DO
              END DO
              CALL DORMQR('Left', 'No transpose', m, nrhs, n, a, lda,       &
                work(1), b, ldb, work(mn+1), lwork-mn, info)
              scllen = m
            END IF
          ELSE
            CALL DGELQF(m, n, a, lda, work(1), work(mn+1), lwork-mn, info)
            IF (.NOT. tpsd) THEN
              CALL DTRTRS('Lower', 'No transpose', 'Non-unit', m, nrhs, a,  &
                lda, b, ldb, info)
              IF (info>0) THEN
                RETURN
              END IF
              DO j = 1, nrhs
                DO i = m + 1, n
                  b(i, j) = zero
                END DO
              END DO
              CALL DORMLQ('Left', 'Transpose', n, nrhs, m, a, lda, work(1), &
                b, ldb, work(mn+1), lwork-mn, info)
              scllen = n
            ELSE
              CALL DORMLQ('Left', 'No transpose', n, nrhs, m, a, lda,       &
                work(1), b, ldb, work(mn+1), lwork-mn, info)
              CALL DTRTRS('Lower', 'Transpose', 'Non-unit', m, nrhs, a,     &
                lda, b, ldb, info)
              IF (info>0) THEN
                RETURN
              END IF
              scllen = m
            END IF
          END IF
          IF (iascl==1) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, smlnum, scllen, nrhs, b,   &
              ldb, info)
          ELSE IF (iascl==2) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, bignum, scllen, nrhs, b,   &
              ldb, info)
          END IF
          IF (ibscl==1) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, smlnum, bnrm, scllen, nrhs, b,   &
              ldb, info)
          ELSE IF (ibscl==2) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, bignum, bnrm, scllen, nrhs, b,   &
              ldb, info)
          END IF
 50       CONTINUE
          work(1) = REAL(wsize,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DGELSD(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, &
          lwork, iwork, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, lda, ldb, lwork, m, n, nrhs, rank
          REAL(rp_) :: rcond
          INTEGER(ip_) :: iwork(*)
          REAL(rp_) :: a(lda, *), b(ldb, *), s(*), work(*)
          REAL(rp_) :: zero, one, two
          PARAMETER (zero=0.0_rp_, one=1.0_rp_, two=2.0_rp_)
          LOGICAL :: lquery
          INTEGER(ip_) :: iascl, ibscl, ie, il, itau, itaup, itauq, ldwork, &
            liwork, maxmn, maxwrk, minmn, minwrk, mm, mnthr, nlvl, nwork,   &
            smlsiz, wlalsd
          REAL(rp_) :: anrm, bignum, bnrm, eps, sfmin, smlnum
          EXTERNAL :: DGEBRD, DGELQF, DGEQRF, DLABAD, &
            DLACPY, DLALSD, DLASCL, DLASET, DORMBR, DORMLQ, &
            DORMQR, XERBLA2
          INTEGER(ip_) :: ILAENV2
          REAL(rp_) :: DLAMCH, DLANGE
          EXTERNAL :: ILAENV2, DLAMCH, DLANGE
          INTRINSIC :: REAL, INT, LOG, MAX, MIN
          info = 0
          minmn = MIN(m, n)
          maxmn = MAX(m, n)
          mnthr = ILAENV2(6_ip_, 'GELSD', ' ', m, n, nrhs, -1_ip_)
          lquery = (lwork==-1)
          IF (m<0) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (nrhs<0) THEN
            info = -3
          ELSE IF (lda<MAX(1,m)) THEN
            info = -5
          ELSE IF (ldb<MAX(1,maxmn)) THEN
            info = -7
          END IF
          smlsiz = ILAENV2(9_ip_, 'GELSD', ' ', 0_ip_, 0_ip_, 0_ip_, 0_ip_)
          minwrk = 1
          liwork = 1
          minmn = MAX(1, minmn)
          nlvl = MAX(INT(LOG(REAL(minmn,rp_)/REAL(smlsiz+1,rp_))/LOG(two))+1, &
                      0_ip_)
          IF (info==0) THEN
            maxwrk = 0
            liwork = 3*minmn*nlvl + 11*minmn
            mm = m
            IF (m>=n .AND. m>=mnthr) THEN
              mm = n
              maxwrk = MAX(maxwrk, n+n*ILAENV2(1_ip_,'GEQRF',' ',m,n,       &
                -1_ip_,-1_ip_))
              maxwrk = MAX(maxwrk, n+nrhs*ILAENV2(1_ip_,'ORMQR','LT',m,     &
                nrhs,n, -1_ip_))
            END IF
            IF (m>=n) THEN
              maxwrk = MAX(maxwrk, 3*n+(mm+n)*ILAENV2(1_ip_,'GEBRD',' ',mm, &
                n, -1_ip_,-1_ip_))
              maxwrk = MAX(maxwrk, 3*n+nrhs*ILAENV2(1_ip_,'ORMBR','QLT',mm, &
                nrhs,n,-1_ip_))
              maxwrk = MAX(maxwrk, 3*n+(n-1)*ILAENV2(1_ip_,'ORMBR','PLN',n, &
                nrhs,n,-1_ip_))
              wlalsd = 9*n + 2*n*smlsiz + 8*n*nlvl + n*nrhs + (smlsiz+1)**2
              maxwrk = MAX(maxwrk, 3*n+wlalsd)
              minwrk = MAX(3*n+mm, 3*n+nrhs, 3*n+wlalsd)
            END IF
            IF (n>m) THEN
              wlalsd = 9*m + 2*m*smlsiz + 8*m*nlvl + m*nrhs + (smlsiz+1)**2
              IF (n>=mnthr) THEN
                maxwrk = m + m*ILAENV2(1_ip_, 'GELQF', ' ', m, n, -1_ip_,   &
                  -1_ip_)
                maxwrk = MAX(maxwrk, m*m+4*m+2*m*ILAENV2(1_ip_,'GEBRD',' ', &
                  m,m ,-1_ip_,-1_ip_))
                maxwrk = MAX(maxwrk, m*m+4*m+nrhs*ILAENV2(1_ip_,'ORMBR',    &
                  'QLT', m,nrhs,m,-1_ip_))
                maxwrk = MAX(maxwrk, m*m+4*m+(m-1)*ILAENV2(1_ip_,'ORMBR',   &
                  'PLN' ,m,nrhs,m,-1_ip_))
                IF (nrhs>1) THEN
                  maxwrk = MAX(maxwrk, m*m+m+m*nrhs)
                ELSE
                  maxwrk = MAX(maxwrk, m*m+2*m)
                END IF
                maxwrk = MAX(maxwrk, m+nrhs*ILAENV2(1_ip_,'ORMLQ','LT',n,   &
                  nrhs, m,-1_ip_))
                maxwrk = MAX(maxwrk, m*m+4*m+wlalsd)
                maxwrk = MAX(maxwrk, 4*m+m*m+MAX(m,2*m-4,nrhs,n-3*m))
              ELSE
                maxwrk = 3*m + (n+m)*ILAENV2(1_ip_, 'GEBRD', ' ', m, n,     &
                  -1_ip_, -1_ip_ )
                maxwrk = MAX(maxwrk, 3*m+nrhs*ILAENV2(1_ip_,'ORMBR','QLT',  &
                  m, nrhs,n,-1_ip_))
                maxwrk = MAX(maxwrk, 3*m+m*ILAENV2(1_ip_,'ORMBR','PLN',n,   &
                  nrhs, m,-1_ip_))
                maxwrk = MAX(maxwrk, 3*m+wlalsd)
              END IF
              minwrk = MAX(3*m+nrhs, 3*m+m, 3*m+wlalsd)
            END IF
            minwrk = MIN(minwrk, maxwrk)
            work(1) = REAL(maxwrk,rp_)
            iwork(1) = liwork
            IF (lwork<minwrk .AND. .NOT. lquery) THEN
              info = -12
            END IF
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GELSD', -info)
            RETURN
          ELSE IF (lquery) THEN
            GO TO 10
          END IF
          IF (m==0 .OR. n==0) THEN
            rank = 0
            RETURN
          END IF
          eps = DLAMCH('P')
          sfmin = DLAMCH('S')
          smlnum = sfmin/eps
          bignum = one/smlnum
          CALL DLABAD(smlnum, bignum)
          anrm = DLANGE('M', m, n, a, lda, work)
          iascl = 0
          IF (anrm>zero .AND. anrm<smlnum) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, smlnum, m, n, a, lda,      &
              info)
            iascl = 1
          ELSE IF (anrm>bignum) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, bignum, m, n, a, lda,      &
              info)
            iascl = 2
          ELSE IF (anrm==zero) THEN
            CALL DLASET('F', MAX(m,n), nrhs, zero, zero, b, ldb)
            CALL DLASET('F', minmn, 1_ip_, zero, zero, s, 1_ip_)
            rank = 0
            GO TO 10
          END IF
          bnrm = DLANGE('M', m, nrhs, b, ldb, work)
          ibscl = 0
          IF (bnrm>zero .AND. bnrm<smlnum) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, bnrm, smlnum, m, nrhs, b, ldb,   &
              info)
            ibscl = 1
          ELSE IF (bnrm>bignum) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, bnrm, bignum, m, nrhs, b, ldb,   &
              info)
            ibscl = 2
          END IF
          IF (m<n) CALL DLASET('F', n-m, nrhs, zero, zero, b(m+1,1), ldb)
          IF (m>=n) THEN
            mm = m
            IF (m>=mnthr) THEN
              mm = n
              itau = 1
              nwork = itau + n
              CALL DGEQRF(m, n, a, lda, work(itau), work(nwork),            &
                lwork-nwork+1, info)
              CALL DORMQR('L', 'T', m, nrhs, n, a, lda, work(itau), b, ldb, &
                work(nwork), lwork-nwork+1, info)
              IF (n>1) THEN
                CALL DLASET('L', n-1, n-1, zero, zero, a(2,1), lda)
              END IF
            END IF
            ie = 1
            itauq = ie + n
            itaup = itauq + n
            nwork = itaup + n
            CALL DGEBRD(mm, n, a, lda, s, work(ie), work(itauq),            &
              work(itaup), work(nwork), lwork-nwork+1, info)
            CALL DORMBR('Q', 'L', 'T', mm, nrhs, n, a, lda, work(itauq), b, &
              ldb, work(nwork), lwork-nwork+1, info)
            CALL DLALSD('U', smlsiz, n, nrhs, s, work(ie), b, ldb, rcond,   &
              rank, work(nwork), iwork, info)
            IF (info/=0) THEN
              GO TO 10
            END IF
            CALL DORMBR('P', 'L', 'N', n, nrhs, n, a, lda, work(itaup), b,  &
              ldb, work(nwork), lwork-nwork+1, info)
          ELSE IF (n>=mnthr .AND. lwork>=4*m+m*m+MAX(m,2*m-4,nrhs,n-3*m,    &
            wlalsd)) THEN
            ldwork = m
            IF (lwork>=MAX(4*m+m*lda+MAX(m,2*m-4,nrhs, n-3*m),              &
              m*lda+m+m*nrhs,4*m+m*lda+wlalsd)) ldwork = lda
            itau = 1
            nwork = m + 1
            CALL DGELQF(m, n, a, lda, work(itau), work(nwork),              &
              lwork-nwork+1, info)
            il = nwork
            CALL DLACPY('L', m, m, a, lda, work(il), ldwork)
            CALL DLASET('U', m-1, m-1, zero, zero, work(il+ldwork), ldwork)
            ie = il + ldwork*m
            itauq = ie + m
            itaup = itauq + m
            nwork = itaup + m
            CALL DGEBRD(m, m, work(il), ldwork, s, work(ie), work(itauq),   &
              work(itaup), work(nwork), lwork-nwork+1, info)
            CALL DORMBR('Q', 'L', 'T', m, nrhs, m, work(il), ldwork,        &
              work(itauq), b, ldb, work(nwork), lwork-nwork+1, info)
            CALL DLALSD('U', smlsiz, m, nrhs, s, work(ie), b, ldb, rcond,   &
              rank, work(nwork), iwork, info)
            IF (info/=0) THEN
              GO TO 10
            END IF
            CALL DORMBR('P', 'L', 'N', m, nrhs, m, work(il), ldwork,        &
              work(itaup), b, ldb, work(nwork), lwork-nwork+1, info)
            CALL DLASET('F', n-m, nrhs, zero, zero, b(m+1,1), ldb)
            nwork = itau + m
            CALL DORMLQ('L', 'T', n, nrhs, m, a, lda, work(itau), b, ldb,   &
              work(nwork), lwork-nwork+1, info)
          ELSE
            ie = 1
            itauq = ie + m
            itaup = itauq + m
            nwork = itaup + m
            CALL DGEBRD(m, n, a, lda, s, work(ie), work(itauq),             &
              work(itaup), work(nwork), lwork-nwork+1, info)
            CALL DORMBR('Q', 'L', 'T', m, nrhs, n, a, lda, work(itauq), b,  &
              ldb, work(nwork), lwork-nwork+1, info)
            CALL DLALSD('L', smlsiz, m, nrhs, s, work(ie), b, ldb, rcond,   &
              rank, work(nwork), iwork, info)
            IF (info/=0) THEN
              GO TO 10
            END IF
            CALL DORMBR('P', 'L', 'N', n, nrhs, m, a, lda, work(itaup), b,  &
              ldb, work(nwork), lwork-nwork+1, info)
          END IF
          IF (iascl==1) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, smlnum, n, nrhs, b, ldb,   &
              info)
            CALL DLASCL('G', 0_ip_, 0_ip_, smlnum, anrm, minmn, 1_ip_, s,   &
              minmn, info)
          ELSE IF (iascl==2) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, bignum, n, nrhs, b, ldb,   &
              info)
            CALL DLASCL('G', 0_ip_, 0_ip_, bignum, anrm, minmn, 1_ip_, s,   &
              minmn, info)
          END IF
          IF (ibscl==1) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, smlnum, bnrm, n, nrhs, b, ldb,   &
              info)
          ELSE IF (ibscl==2) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, bignum, bnrm, n, nrhs, b, ldb,   &
              info)
          END IF
 10       CONTINUE
          work(1) = REAL(maxwrk,rp_)
          iwork(1) = liwork
          RETURN
        END SUBROUTINE

        SUBROUTINE DGELSS(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, &
          lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, lda, ldb, lwork, m, n, nrhs, rank
          REAL(rp_) :: rcond
          REAL(rp_) :: a(lda, *), b(ldb, *), s(*), work(*)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          LOGICAL :: lquery
          INTEGER(ip_) :: bdspac, bl, chunk, i, iascl, ibscl, ie, il, itau, &
            itaup, itauq, iwork, ldwork, maxmn, maxwrk, minmn, minwrk, mm,  &
            mnthr
          INTEGER(ip_) :: lwork_dgeqrf, lwork_dormqr, lwork_dgebrd,         &
            lwork_dormbr, lwork_dorgbr, lwork_dormlq, lwork_dgelqf
          REAL(rp_) :: anrm, bignum, bnrm, eps, sfmin, smlnum, thr
          REAL(rp_) :: dum(1)
          EXTERNAL :: DBDSQR, DCOPY, DGEBRD, DGELQF, DGEMM,& 
            DGEMV, DGEQRF, DLABAD, DLACPY, DLASCL, DLASET, &
            DORGBR, DORMBR, DORMLQ, DORMQR, DRSCL, XERBLA2
          INTEGER(ip_) :: ILAENV2
          REAL(rp_) :: DLAMCH, DLANGE
          EXTERNAL :: ILAENV2, DLAMCH, DLANGE
          INTRINSIC :: MAX, MIN
          info = 0
          minmn = MIN(m, n)
          maxmn = MAX(m, n)
          lquery = (lwork==-1)
          IF (m<0) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (nrhs<0) THEN
            info = -3
          ELSE IF (lda<MAX(1,m)) THEN
            info = -5
          ELSE IF (ldb<MAX(1,maxmn)) THEN
            info = -7
          END IF
          IF (info==0) THEN
            minwrk = 1
            maxwrk = 1
            IF (minmn>0) THEN
              mm = m
              mnthr = ILAENV2(6_ip_, 'GELSS', ' ', m, n, nrhs, -1_ip_)
              IF (m>=n .AND. m>=mnthr) THEN
                CALL DGEQRF(m, n, a, lda, dum(1), dum(1), -1_ip_, info)
                lwork_dgeqrf = INT(dum(1),ip_)
                CALL DORMQR('L', 'T', m, nrhs, n, a, lda, dum(1), b, ldb,   &
                  dum(1), -1_ip_, info)
                lwork_dormqr = INT(dum(1),ip_)
                mm = n
                maxwrk = MAX(maxwrk, n+lwork_dgeqrf)
                maxwrk = MAX(maxwrk, n+lwork_dormqr)
              END IF
              IF (m>=n) THEN
                bdspac = MAX(1, 5*n)
                CALL DGEBRD(mm, n, a, lda, s, dum(1), dum(1), dum(1),       &
                  dum(1), -1_ip_, info)
                lwork_dgebrd = INT(dum(1),ip_)
                CALL DORMBR('Q', 'L', 'T', mm, nrhs, n, a, lda, dum(1), b,  &
                  ldb, dum(1), -1_ip_, info)
                lwork_dormbr = INT(dum(1),ip_)
                CALL DORGBR('P', n, n, n, a, lda, dum(1), dum(1), -1_ip_,   &
                  info)
                lwork_dorgbr = INT(dum(1),ip_)
                maxwrk = MAX(maxwrk, 3*n+lwork_dgebrd)
                maxwrk = MAX(maxwrk, 3*n+lwork_dormbr)
                maxwrk = MAX(maxwrk, 3*n+lwork_dorgbr)
                maxwrk = MAX(maxwrk, bdspac)
                maxwrk = MAX(maxwrk, n*nrhs)
                minwrk = MAX(3*n+mm, 3*n+nrhs, bdspac)
                maxwrk = MAX(minwrk, maxwrk)
              END IF
              IF (n>m) THEN
                bdspac = MAX(1, 5*m)
                minwrk = MAX(3*m+nrhs, 3*m+n, bdspac)
                IF (n>=mnthr) THEN
                  CALL DGELQF(m, n, a, lda, dum(1), dum(1), -1_ip_, info)
                  lwork_dgelqf = INT(dum(1),ip_)
                  CALL DGEBRD(m, m, a, lda, s, dum(1), dum(1), dum(1),      &
                    dum(1), -1_ip_, info)
                  lwork_dgebrd = INT(dum(1),ip_)
                  CALL DORMBR('Q', 'L', 'T', m, nrhs, n, a, lda, dum(1), b, &
                    ldb, dum(1), -1_ip_, info)
                  lwork_dormbr = INT(dum(1),ip_)
                  CALL DORGBR('P', m, m, m, a, lda, dum(1), dum(1), -1_ip_, &
                    info)
                  lwork_dorgbr = INT(dum(1),ip_)
                  CALL DORMLQ('L', 'T', n, nrhs, m, a, lda, dum(1), b, ldb, &
                    dum(1), -1_ip_, info)
                  lwork_dormlq = INT(dum(1),ip_)
                  maxwrk = m + lwork_dgelqf
                  maxwrk = MAX(maxwrk, m*m+4*m+lwork_dgebrd)
                  maxwrk = MAX(maxwrk, m*m+4*m+lwork_dormbr)
                  maxwrk = MAX(maxwrk, m*m+4*m+lwork_dorgbr)
                  maxwrk = MAX(maxwrk, m*m+m+bdspac)
                  IF (nrhs>1) THEN
                    maxwrk = MAX(maxwrk, m*m+m+m*nrhs)
                  ELSE
                    maxwrk = MAX(maxwrk, m*m+2*m)
                  END IF
                  maxwrk = MAX(maxwrk, m+lwork_dormlq)
                ELSE
                  CALL DGEBRD(m, n, a, lda, s, dum(1), dum(1), dum(1),      &
                    dum(1), -1_ip_, info)
                  lwork_dgebrd = INT(dum(1),ip_)
                  CALL DORMBR('Q', 'L', 'T', m, nrhs, m, a, lda, dum(1), b, &
                    ldb, dum(1), -1_ip_, info)
                  lwork_dormbr = INT(dum(1),ip_)
                  CALL DORGBR('P', m, n, m, a, lda, dum(1), dum(1), -1_ip_, &
                    info)
                  lwork_dorgbr = INT(dum(1),ip_)
                  maxwrk = 3*m + lwork_dgebrd
                  maxwrk = MAX(maxwrk, 3*m+lwork_dormbr)
                  maxwrk = MAX(maxwrk, 3*m+lwork_dorgbr)
                  maxwrk = MAX(maxwrk, bdspac)
                  maxwrk = MAX(maxwrk, n*nrhs)
                END IF
              END IF
              maxwrk = MAX(minwrk, maxwrk)
            END IF
            work(1) = REAL(maxwrk,rp_)
            IF (lwork<minwrk .AND. .NOT. lquery) info = -12
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GELSS', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          IF (m==0 .OR. n==0) THEN
            rank = 0
            RETURN
          END IF
          eps = DLAMCH('P')
          sfmin = DLAMCH('S')
          smlnum = sfmin/eps
          bignum = one/smlnum
          CALL DLABAD(smlnum, bignum)
          anrm = DLANGE('M', m, n, a, lda, work)
          iascl = 0
          IF (anrm>zero .AND. anrm<smlnum) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, smlnum, m, n, a, lda,      &
              info)
            iascl = 1
          ELSE IF (anrm>bignum) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, bignum, m, n, a, lda,      &
              info)
            iascl = 2
          ELSE IF (anrm==zero) THEN
            CALL DLASET('F', MAX(m,n), nrhs, zero, zero, b, ldb)
            CALL DLASET('F', minmn, 1_ip_, zero, zero, s, minmn)
            rank = 0
            GO TO 70
          END IF
          bnrm = DLANGE('M', m, nrhs, b, ldb, work)
          ibscl = 0
          IF (bnrm>zero .AND. bnrm<smlnum) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, bnrm, smlnum, m, nrhs, b, ldb,   &
              info)
            ibscl = 1
          ELSE IF (bnrm>bignum) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, bnrm, bignum, m, nrhs, b, ldb,   &
              info)
            ibscl = 2
          END IF
          IF (m>=n) THEN
            mm = m
            IF (m>=mnthr) THEN
              mm = n
              itau = 1
              iwork = itau + n
              CALL DGEQRF(m, n, a, lda, work(itau), work(iwork),            &
                lwork-iwork+1, info)
              CALL DORMQR('L', 'T', m, nrhs, n, a, lda, work(itau), b, ldb, &
                work(iwork), lwork-iwork+1, info)
              IF (n>1) CALL DLASET('L', n-1, n-1, zero, zero, a(2,1), lda)
            END IF
            ie = 1
            itauq = ie + n
            itaup = itauq + n
            iwork = itaup + n
            CALL DGEBRD(mm, n, a, lda, s, work(ie), work(itauq),            &
              work(itaup), work(iwork), lwork-iwork+1, info)
            CALL DORMBR('Q', 'L', 'T', mm, nrhs, n, a, lda, work(itauq), b, &
              ldb, work(iwork), lwork-iwork+1, info)
            CALL DORGBR('P', n, n, n, a, lda, work(itaup), work(iwork),     &
              lwork-iwork+1, info)
            iwork = ie + n
            CALL DBDSQR('U', n, n, 0_ip_, nrhs, s, work(ie), a, lda, dum,   &
              1_ip_, b, ldb, work(iwork), info)
            IF (info/=0) GO TO 70
            thr = MAX(rcond*s(1), sfmin)
            IF (rcond<zero) thr = MAX(eps*s(1), sfmin)
            rank = 0
            DO i = 1, n
              IF (s(i)>thr) THEN
                CALL DRSCL(nrhs, s(i), b(i,1), ldb)
                rank = rank + 1
              ELSE
                CALL DLASET('F', 1_ip_, nrhs, zero, zero, b(i,1), ldb)
              END IF
            END DO
            IF (lwork>=ldb*nrhs .AND. nrhs>1) THEN
              CALL DGEMM('T', 'N', n, nrhs, n, one, a, lda, b, ldb, zero,   &
                work, ldb)
              CALL DLACPY('G', n, nrhs, work, ldb, b, ldb)
            ELSE IF (nrhs>1) THEN
              chunk = lwork/n
              DO i = 1, nrhs, chunk
                bl = MIN(nrhs-i+1, chunk)
                CALL DGEMM('T', 'N', n, bl, n, one, a, lda, b(1,i), ldb,    &
                  zero, work, n)
                CALL DLACPY('G', n, bl, work, n, b(1,i), ldb)
              END DO
            ELSE
              CALL DGEMV('T', n, n, one, a, lda, b, 1_ip_, zero, work,      &
                1_ip_)
              CALL DCOPY(n, work, 1_ip_, b, 1_ip_)
            END IF
          ELSE IF (n>=mnthr .AND. lwork>=4*m+m*m+MAX(m,2*m-4,nrhs,n-3*m))   &
            THEN
            ldwork = m
            IF (lwork>=MAX(4*m+m*lda+MAX(m,2*m-4,nrhs, n-3*m),              &
              m*lda+m+m*nrhs)) ldwork = lda
            itau = 1
            iwork = m + 1
            CALL DGELQF(m, n, a, lda, work(itau), work(iwork),              &
              lwork-iwork+1, info)
            il = iwork
            CALL DLACPY('L', m, m, a, lda, work(il), ldwork)
            CALL DLASET('U', m-1, m-1, zero, zero, work(il+ldwork), ldwork)
            ie = il + ldwork*m
            itauq = ie + m
            itaup = itauq + m
            iwork = itaup + m
            CALL DGEBRD(m, m, work(il), ldwork, s, work(ie), work(itauq),   &
              work(itaup), work(iwork), lwork-iwork+1, info)
            CALL DORMBR('Q', 'L', 'T', m, nrhs, m, work(il), ldwork,        &
              work(itauq), b, ldb, work(iwork), lwork-iwork+1, info)
            CALL DORGBR('P', m, m, m, work(il), ldwork, work(itaup),        &
              work(iwork), lwork-iwork+1, info)
            iwork = ie + m
            CALL DBDSQR('U', m, m, 0_ip_, nrhs, s, work(ie), work(il),      &
              ldwork, a, lda, b, ldb, work(iwork), info)
            IF (info/=0) GO TO 70
            thr = MAX(rcond*s(1), sfmin)
            IF (rcond<zero) thr = MAX(eps*s(1), sfmin)
            rank = 0
            DO i = 1, m
              IF (s(i)>thr) THEN
                CALL DRSCL(nrhs, s(i), b(i,1), ldb)
                rank = rank + 1
              ELSE
                CALL DLASET('F', 1_ip_, nrhs, zero, zero, b(i,1), ldb)
              END IF
            END DO
            iwork = ie
            IF (lwork>=ldb*nrhs+iwork-1 .AND. nrhs>1) THEN
              CALL DGEMM('T', 'N', m, nrhs, m, one, work(il), ldwork, b,    &
                ldb, zero, work(iwork), ldb)
              CALL DLACPY('G', m, nrhs, work(iwork), ldb, b, ldb)
            ELSE IF (nrhs>1) THEN
              chunk = (lwork-iwork+1)/m
              DO i = 1, nrhs, chunk
                bl = MIN(nrhs-i+1, chunk)
                CALL DGEMM('T', 'N', m, bl, m, one, work(il), ldwork, b(1,  &
                  i), ldb, zero, work(iwork), m)
                CALL DLACPY('G', m, bl, work(iwork), m, b(1,i), ldb)
              END DO
            ELSE
              CALL DGEMV('T', m, m, one, work(il), ldwork, b(1,1), 1_ip_,   &
                zero, work(iwork), 1_ip_)
              CALL DCOPY(m, work(iwork), 1_ip_, b(1,1), 1_ip_)
            END IF
            CALL DLASET('F', n-m, nrhs, zero, zero, b(m+1,1), ldb)
            iwork = itau + m
            CALL DORMLQ('L', 'T', n, nrhs, m, a, lda, work(itau), b, ldb,   &
              work(iwork), lwork-iwork+1, info)
          ELSE
            ie = 1
            itauq = ie + m
            itaup = itauq + m
            iwork = itaup + m
            CALL DGEBRD(m, n, a, lda, s, work(ie), work(itauq),             &
              work(itaup), work(iwork), lwork-iwork+1, info)
            CALL DORMBR('Q', 'L', 'T', m, nrhs, n, a, lda, work(itauq), b,  &
              ldb, work(iwork), lwork-iwork+1, info)
            CALL DORGBR('P', m, n, m, a, lda, work(itaup), work(iwork),     &
              lwork-iwork+1, info)
            iwork = ie + m
            CALL DBDSQR('L', m, n, 0_ip_, nrhs, s, work(ie), a, lda, dum,   &
              1_ip_, b, ldb, work(iwork), info)
            IF (info/=0) GO TO 70
            thr = MAX(rcond*s(1), sfmin)
            IF (rcond<zero) thr = MAX(eps*s(1), sfmin)
            rank = 0
            DO i = 1, m
              IF (s(i)>thr) THEN
                CALL DRSCL(nrhs, s(i), b(i,1), ldb)
                rank = rank + 1
              ELSE
                CALL DLASET('F', 1_ip_, nrhs, zero, zero, b(i,1), ldb)
              END IF
            END DO
            IF (lwork>=ldb*nrhs .AND. nrhs>1) THEN
              CALL DGEMM('T', 'N', n, nrhs, m, one, a, lda, b, ldb, zero,   &
                work, ldb)
              CALL DLACPY('F', n, nrhs, work, ldb, b, ldb)
            ELSE IF (nrhs>1) THEN
              chunk = lwork/n
              DO i = 1, nrhs, chunk
                bl = MIN(nrhs-i+1, chunk)
                CALL DGEMM('T', 'N', n, bl, m, one, a, lda, b(1,i), ldb,    &
                  zero, work, n)
                CALL DLACPY('F', n, bl, work, n, b(1,i), ldb)
              END DO
            ELSE
              CALL DGEMV('T', m, n, one, a, lda, b, 1_ip_, zero, work,      &
                1_ip_)
              CALL DCOPY(n, work, 1_ip_, b, 1_ip_)
            END IF
          END IF
          IF (iascl==1) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, smlnum, n, nrhs, b, ldb,   &
              info)
            CALL DLASCL('G', 0_ip_, 0_ip_, smlnum, anrm, minmn, 1_ip_, s,   &
              minmn, info)
          ELSE IF (iascl==2) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, bignum, n, nrhs, b, ldb,   &
              info)
            CALL DLASCL('G', 0_ip_, 0_ip_, bignum, anrm, minmn, 1_ip_, s,   &
              minmn, info)
          END IF
          IF (ibscl==1) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, smlnum, bnrm, n, nrhs, b, ldb,   &
              info)
          ELSE IF (ibscl==2) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, bignum, bnrm, n, nrhs, b, ldb,   &
              info)
          END IF
 70       CONTINUE
          work(1) = REAL(maxwrk,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DGELSY(m, n, nrhs, a, lda, b, ldb, jpvt, rcond, rank,    &
          work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, lda, ldb, lwork, m, n, nrhs, rank
          REAL(rp_) :: rcond
          INTEGER(ip_) :: jpvt(*)
          REAL(rp_) :: a(lda, *), b(ldb, *), work(*)
          INTEGER(ip_) :: imax, imin
          PARAMETER (imax=1, imin=2)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          LOGICAL :: lquery
          INTEGER(ip_) :: i, iascl, ibscl, ismax, ismin, j, lwkmin, lwkopt, &
            mn, nb, nb1, nb2, nb3, nb4
          REAL(rp_) :: anrm, bignum, bnrm, c1, c2, s1, s2, smax, smaxpr,    &
            smin, sminpr, smlnum, wsize
          INTEGER(ip_) :: ILAENV2
          REAL(rp_) :: DLAMCH, DLANGE
          EXTERNAL :: ILAENV2, DLAMCH, DLANGE
          EXTERNAL :: DCOPY, DGEQP3, DLABAD, DLAIC1, &
            DLASCL, DLASET, DORMQR, DORMRZ, DTRSM, DTZRZF, &
            XERBLA2
          INTRINSIC :: ABS, MAX, MIN
          mn = MIN(m, n)
          ismin = mn + 1
          ismax = 2*mn + 1
          info = 0
          lquery = (lwork==-1)
          IF (m<0) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (nrhs<0) THEN
            info = -3
          ELSE IF (lda<MAX(1,m)) THEN
            info = -5
          ELSE IF (ldb<MAX(1,m,n)) THEN
            info = -7
          END IF
          IF (info==0) THEN
            IF (mn==0 .OR. nrhs==0) THEN
              lwkmin = 1
              lwkopt = 1
            ELSE
              nb1 = ILAENV2(1_ip_, 'GEQRF', ' ', m, n, -1_ip_, -1_ip_)
              nb2 = ILAENV2(1_ip_, 'GERQF', ' ', m, n, -1_ip_, -1_ip_)
              nb3 = ILAENV2(1_ip_, 'ORMQR', ' ', m, n, nrhs, -1_ip_)
              nb4 = ILAENV2(1_ip_, 'ORMRQ', ' ', m, n, nrhs, -1_ip_)
              nb = MAX(nb1, nb2, nb3, nb4)
              lwkmin = mn + MAX(2*mn, n+1, mn+nrhs)
              lwkopt = MAX(lwkmin, mn+2*n+nb*(n+1), 2*mn+nb*nrhs)
            END IF
            work(1) = REAL(lwkopt,rp_)
            IF (lwork<lwkmin .AND. .NOT. lquery) THEN
              info = -12
            END IF
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GELSY', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          IF (mn==0 .OR. nrhs==0) THEN
            rank = 0
            RETURN
          END IF
          smlnum = DLAMCH('S')/DLAMCH('P')
          bignum = one/smlnum
          CALL DLABAD(smlnum, bignum)
          anrm = DLANGE('M', m, n, a, lda, work)
          iascl = 0
          IF (anrm>zero .AND. anrm<smlnum) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, smlnum, m, n, a, lda,      &
              info)
            iascl = 1
          ELSE IF (anrm>bignum) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, bignum, m, n, a, lda,      &
              info)
            iascl = 2
          ELSE IF (anrm==zero) THEN
            CALL DLASET('F', MAX(m,n), nrhs, zero, zero, b, ldb)
            rank = 0
            GO TO 70
          END IF
          bnrm = DLANGE('M', m, nrhs, b, ldb, work)
          ibscl = 0
          IF (bnrm>zero .AND. bnrm<smlnum) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, bnrm, smlnum, m, nrhs, b, ldb,   &
              info)
            ibscl = 1
          ELSE IF (bnrm>bignum) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, bnrm, bignum, m, nrhs, b, ldb,   &
              info)
            ibscl = 2
          END IF
          CALL DGEQP3(m, n, a, lda, jpvt, work(1), work(mn+1), lwork-mn,    &
            info)
          wsize = REAL(mn,rp_) + work(mn+1)
          work(ismin) = one
          work(ismax) = one
          smax = ABS(a(1,1))
          smin = smax
          IF (ABS(a(1,1))==zero) THEN
            rank = 0
            CALL DLASET('F', MAX(m,n), nrhs, zero, zero, b, ldb)
            GO TO 70
          ELSE
            rank = 1
          END IF
 10       CONTINUE
          IF (rank<mn) THEN
            i = rank + 1
            CALL DLAIC1(imin, rank, work(ismin), smin, a(1,i), a(i,i),      &
              sminpr, s1, c1)
            CALL DLAIC1(imax, rank, work(ismax), smax, a(1,i), a(i,i),      &
              smaxpr, s2, c2)
            IF (smaxpr*rcond<=sminpr) THEN
              DO i = 1, rank
                work(ismin+i-1) = s1*work(ismin+i-1)
                work(ismax+i-1) = s2*work(ismax+i-1)
              END DO
              work(ismin+rank) = c1
              work(ismax+rank) = c2
              smin = sminpr
              smax = smaxpr
              rank = rank + 1
              GO TO 10
            END IF
          END IF
          IF (rank<n) CALL DTZRZF(rank, n, a, lda, work(mn+1),              &
            work(2*mn+1), lwork-2*mn, info)
          CALL DORMQR('Left', 'Transpose', m, nrhs, mn, a, lda, work(1), b, &
            ldb, work(2*mn+1), lwork-2*mn, info)
          wsize = MAX(wsize, REAL(2*mn,rp_)+work(2*mn+1))
          CALL DTRSM('Left', 'Upper', 'No transpose', 'Non-unit', rank,     &
            nrhs, one, a, lda, b, ldb)
          DO j = 1, nrhs
            DO i = rank + 1, n
              b(i, j) = zero
            END DO
          END DO
          IF (rank<n) THEN
            CALL DORMRZ('Left', 'Transpose', n, nrhs, rank, n-rank, a, lda, &
              work(mn+1), b, ldb, work(2*mn+1), lwork-2*mn, info)
          END IF
          DO j = 1, nrhs
            DO i = 1, n
              work(jpvt(i)) = b(i, j)
            END DO
            CALL DCOPY(n, work(1), 1_ip_, b(1,j), 1_ip_)
          END DO
          IF (iascl==1) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, smlnum, n, nrhs, b, ldb,   &
              info)
            CALL DLASCL('U', 0_ip_, 0_ip_, smlnum, anrm, rank, rank, a,     &
              lda, info)
          ELSE IF (iascl==2) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, bignum, n, nrhs, b, ldb,   &
              info)
            CALL DLASCL('U', 0_ip_, 0_ip_, bignum, anrm, rank, rank, a,     &
              lda, info)
          END IF
          IF (ibscl==1) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, smlnum, bnrm, n, nrhs, b, ldb,   &
              info)
          ELSE IF (ibscl==2) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, bignum, bnrm, n, nrhs, b, ldb,   &
              info)
          END IF
 70       CONTINUE
          work(1) = REAL(lwkopt,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DGEQP3(m, n, a, lda, jpvt, tau, work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, lda, lwork, m, n
          INTEGER(ip_) :: jpvt(*)
          REAL(rp_) :: a(lda, *), tau(*), work(*)
          INTEGER(ip_) :: inb, inbmin, ixover
          PARAMETER (inb=1, inbmin=2, ixover=3)
          LOGICAL :: lquery
          INTEGER(ip_) :: fjb, iws, j, jb, lwkopt, minmn, minws, na, nb,    &
            nbmin, nfxd, nx, sm, sminmn, sn, topbmn
          EXTERNAL :: DGEQRF, DLAQP2, DLAQPS, DORMQR, &
            DSWAP, XERBLA2
          INTEGER(ip_) :: ILAENV2
          REAL(rp_) :: DNRM2
          EXTERNAL :: ILAENV2, DNRM2
          INTRINSIC :: INT, MAX, MIN
          info = 0
          lquery = (lwork==-1)
          IF (m<0) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,m)) THEN
            info = -4
          END IF
          IF (info==0) THEN
            minmn = MIN(m, n)
            IF (minmn==0) THEN
              iws = 1
              lwkopt = 1
            ELSE
              iws = 3*n + 1
              nb = ILAENV2(inb, 'GEQRF', ' ', m, n, -1_ip_, -1_ip_)
              lwkopt = 2*n + (n+1)*nb
            END IF
            work(1) = REAL(lwkopt,rp_)
            IF ((lwork<iws) .AND. .NOT. lquery) THEN
              info = -8
            END IF
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GEQP3', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          nfxd = 1
          DO j = 1, n
            IF (jpvt(j)/=0) THEN
              IF (j/=nfxd) THEN
                CALL DSWAP(m, a(1,j), 1_ip_, a(1,nfxd), 1_ip_)
                jpvt(j) = jpvt(nfxd)
                jpvt(nfxd) = j
              ELSE
                jpvt(j) = j
              END IF
              nfxd = nfxd + 1
            ELSE
              jpvt(j) = j
            END IF
          END DO
          nfxd = nfxd - 1
          IF (nfxd>0) THEN
            na = MIN(m, nfxd)
            CALL DGEQRF(m, na, a, lda, tau, work, lwork, info)
            iws = MAX(iws, INT(work(1)))
            IF (na<n) THEN
              CALL DORMQR('Left', 'Transpose', m, n-na, na, a, lda, tau,    &
                a(1,na+1), lda, work, lwork, info)
              iws = MAX(iws, INT(work(1)))
            END IF
          END IF
          IF (nfxd<minmn) THEN
            sm = m - nfxd
            sn = n - nfxd
            sminmn = minmn - nfxd
            nb = ILAENV2(inb, 'GEQRF', ' ', sm, sn, -1_ip_, -1_ip_)
            nbmin = 2
            nx = 0
            IF ((nb>1) .AND. (nb<sminmn)) THEN
              nx = MAX(0_ip_, ILAENV2(ixover,'GEQRF',' ',sm,sn,-1_ip_,      &
                -1_ip_))
              IF (nx<sminmn) THEN
                minws = 2*sn + (sn+1)*nb
                iws = MAX(iws, minws)
                IF (lwork<minws) THEN
                  nb = (lwork-2*sn)/(sn+1)
                  nbmin = MAX(2_ip_, ILAENV2(inbmin,'GEQRF',' ',sm,sn,      &
                    -1_ip_,-1_ip_))
                END IF
              END IF
            END IF
            DO j = nfxd + 1, n
              work(j) = DNRM2(sm, a(nfxd+1,j), 1_ip_)
              work(n+j) = work(j)
            END DO
            IF ((nb>=nbmin) .AND. (nb<sminmn) .AND. (nx<sminmn)) THEN
              j = nfxd + 1
              topbmn = minmn - nx
 30           CONTINUE
              IF (j<=topbmn) THEN
                jb = MIN(nb, topbmn-j+1)
                CALL DLAQPS(m, n-j+1, j-1, jb, fjb, a(1,j), lda, jpvt(j),   &
                  tau(j), work(j), work(n+j), work(2*n+1), work(2*n+jb+1),  &
                  n-j+1)
                j = j + fjb
                GO TO 30
              END IF
            ELSE
              j = nfxd + 1
            END IF
            IF (j<=minmn) CALL DLAQP2(m, n-j+1, j-1, a(1,j), lda, jpvt(j),  &
              tau(j), work(j), work(n+j), work(2*n+1))
          END IF
          work(1) = REAL(iws,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DGEQR2(m, n, a, lda, tau, work, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, lda, m, n
          REAL(rp_) :: a(lda, *), tau(*), work(*)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          INTEGER(ip_) :: i, k
          REAL(rp_) :: aii
          EXTERNAL :: DLARF, DLARFG, XERBLA2
          INTRINSIC :: MAX, MIN
          info = 0
          IF (m<0) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,m)) THEN
            info = -4
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GEQR2', -info)
            RETURN
          END IF
          k = MIN(m, n)
          DO i = 1, k
            CALL DLARFG(m-i+1, a(i,i), a(MIN(i+1,m),i), 1_ip_, tau(i))
            IF (i<n) THEN
              aii = a(i, i)
              a(i, i) = one
              CALL DLARF('Left', m-i+1, n-i, a(i,i), 1_ip_, tau(i), a(i,    &
                i+1), lda, work)
              a(i, i) = aii
            END IF
          END DO
          RETURN
        END SUBROUTINE

        SUBROUTINE DGEQRF(m, n, a, lda, tau, work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, lda, lwork, m, n
          REAL(rp_) :: a(lda, *), tau(*), work(*)
          LOGICAL :: lquery
          INTEGER(ip_) :: i, ib, iinfo, iws, k, ldwork, lwkopt, nb, nbmin,  &
            nx
          EXTERNAL :: DGEQR2, DLARFB, DLARFT, XERBLA2
          INTRINSIC :: MAX, MIN
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: ILAENV2
          info = 0
          nb = ILAENV2(1_ip_, 'GEQRF', ' ', m, n, -1_ip_, -1_ip_)
          lwkopt = n*nb
          work(1) = REAL(lwkopt,rp_)
          lquery = (lwork==-1)
          IF (m<0) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,m)) THEN
            info = -4
          ELSE IF (lwork<MAX(1,n) .AND. .NOT. lquery) THEN
            info = -7
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GEQRF', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          k = MIN(m, n)
          IF (k==0) THEN
            work(1) = 1
            RETURN
          END IF
          nbmin = 2
          nx = 0
          iws = n
          IF (nb>1 .AND. nb<k) THEN
            nx = MAX(0_ip_, ILAENV2(3_ip_,'GEQRF',' ',m,n,-1_ip_,-1_ip_))
            IF (nx<k) THEN
              ldwork = n
              iws = ldwork*nb
              IF (lwork<iws) THEN
                nb = lwork/ldwork
                nbmin = MAX(2_ip_, ILAENV2(2_ip_,'GEQRF',' ',m,n,-1_ip_,    &
                  -1_ip_))
              END IF
            END IF
          END IF
          IF (nb>=nbmin .AND. nb<k .AND. nx<k) THEN
            DO i = 1, k - nx, nb
              ib = MIN(k-i+1, nb)
              CALL DGEQR2(m-i+1, ib, a(i,i), lda, tau(i), work, iinfo)
              IF (i+ib<=n) THEN
                CALL DLARFT('Forward', 'Columnwise', m-i+1, ib, a(i,i),     &
                  lda, tau(i), work, ldwork)
                CALL DLARFB('Left', 'Transpose', 'Forward', 'Columnwise',   &
                  m-i+1, n-i-ib+1, ib, a(i,i), lda, work, ldwork, a(i,i+ib),&
                  lda, work(ib+1), ldwork)
              END IF
            END DO
          ELSE
            i = 1
          END IF
          IF (i<=k) CALL DGEQR2(m-i+1, n-i+1, a(i,i), lda, tau(i), work,    &
            iinfo)
          work(1) = REAL(iws,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DGESVD(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,   &
          work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: jobu, jobvt
          INTEGER(ip_) :: info, lda, ldu, ldvt, lwork, m, n
          REAL(rp_) :: a(lda, *), s(*), u(ldu, *), vt(ldvt, *), work(*)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          LOGICAL :: lquery, wntua, wntuas, wntun, wntuo, wntus, wntva,     &
            wntvas, wntvn, wntvo, wntvs
          INTEGER(ip_) :: bdspac, blk, chunk, i, ie, ierr, ir, iscl, itau,  &
            itaup, itauq, iu, iwork, ldwrkr, ldwrku, maxwrk, minmn, minwrk, &
            mnthr, ncu, ncvt, nru, nrvt, wrkbl
          INTEGER(ip_) :: lwork_dgeqrf, lwork_dorgqr_n, lwork_dorgqr_m,     &
            lwork_dgebrd, lwork_dorgbr_p, lwork_dorgbr_q, lwork_dgelqf,     &
            lwork_dorglq_n, lwork_dorglq_m
          REAL(rp_) :: anrm, bignum, eps, smlnum
          REAL(rp_) :: dum(1)
          EXTERNAL :: DBDSQR, DGEBRD, DGELQF, DGEMM, &
            DGEQRF, DLACPY, DLASCL, DLASET, DORGBR, DORGLQ, &
            DORGQR, DORMBR, XERBLA2
          LOGICAL :: LSAME
          INTEGER(ip_) :: ILAENV2
          REAL(rp_) :: DLAMCH, DLANGE
          EXTERNAL :: LSAME, ILAENV2, DLAMCH, DLANGE
          INTRINSIC :: MAX, MIN, SQRT
          info = 0
          minmn = MIN(m, n)
          wntua = LSAME(jobu, 'A')
          wntus = LSAME(jobu, 'S')
          wntuas = wntua .OR. wntus
          wntuo = LSAME(jobu, 'O')
          wntun = LSAME(jobu, 'N')
          wntva = LSAME(jobvt, 'A')
          wntvs = LSAME(jobvt, 'S')
          wntvas = wntva .OR. wntvs
          wntvo = LSAME(jobvt, 'O')
          wntvn = LSAME(jobvt, 'N')
          lquery = (lwork==-1)
          IF (.NOT. (wntua .OR. wntus .OR. wntuo .OR. wntun)) THEN
            info = -1
          ELSE IF (.NOT. (wntva .OR. wntvs .OR. wntvo .OR. wntvn) .OR.      &
            (wntvo .AND. wntuo)) THEN
            info = -2
          ELSE IF (m<0) THEN
            info = -3
          ELSE IF (n<0) THEN
            info = -4
          ELSE IF (lda<MAX(1,m)) THEN
            info = -6
          ELSE IF (ldu<1 .OR. (wntuas .AND. ldu<m)) THEN
            info = -9
          ELSE IF (ldvt<1 .OR. (wntva .AND. ldvt<n) .OR. (wntvs .AND.       &
            ldvt<minmn)) THEN
            info = -11
          END IF
          IF (info==0) THEN
            minwrk = 1
            maxwrk = 1
            IF (m>=n .AND. minmn>0) THEN
              mnthr = ILAENV2(6_ip_, 'GESVD', jobu//jobvt, m, n, 0_ip_,     &
                0_ip_)
              bdspac = 5*n
              CALL DGEQRF(m, n, a, lda, dum(1), dum(1), -1_ip_, ierr)
              lwork_dgeqrf = INT(dum(1))
              CALL DORGQR(m, n, n, a, lda, dum(1), dum(1), -1_ip_, ierr)
              lwork_dorgqr_n = INT(dum(1))
              CALL DORGQR(m, m, n, a, lda, dum(1), dum(1), -1_ip_, ierr)
              lwork_dorgqr_m = INT(dum(1))
              CALL DGEBRD(n, n, a, lda, s, dum(1), dum(1), dum(1), dum(1),  &
                -1_ip_, ierr)
              lwork_dgebrd = INT(dum(1))
              CALL DORGBR('P', n, n, n, a, lda, dum(1), dum(1), -1_ip_,     &
                ierr)
              lwork_dorgbr_p = INT(dum(1))
              CALL DORGBR('Q', n, n, n, a, lda, dum(1), dum(1), -1_ip_,     &
                ierr)
              lwork_dorgbr_q = INT(dum(1))
              IF (m>=mnthr) THEN
                IF (wntun) THEN
                  maxwrk = n + lwork_dgeqrf
                  maxwrk = MAX(maxwrk, 3*n+lwork_dgebrd)
                  IF (wntvo .OR. wntvas) maxwrk = MAX(maxwrk,               &
                    3*n+lwork_dorgbr_p)
                  maxwrk = MAX(maxwrk, bdspac)
                  minwrk = MAX(4*n, bdspac)
                ELSE IF (wntuo .AND. wntvn) THEN
                  wrkbl = n + lwork_dgeqrf
                  wrkbl = MAX(wrkbl, n+lwork_dorgqr_n)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dgebrd)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dorgbr_q)
                  wrkbl = MAX(wrkbl, bdspac)
                  maxwrk = MAX(n*n+wrkbl, n*n+m*n+n)
                  minwrk = MAX(3*n+m, bdspac)
                ELSE IF (wntuo .AND. wntvas) THEN
                  wrkbl = n + lwork_dgeqrf
                  wrkbl = MAX(wrkbl, n+lwork_dorgqr_n)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dgebrd)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dorgbr_q)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dorgbr_p)
                  wrkbl = MAX(wrkbl, bdspac)
                  maxwrk = MAX(n*n+wrkbl, n*n+m*n+n)
                  minwrk = MAX(3*n+m, bdspac)
                ELSE IF (wntus .AND. wntvn) THEN
                  wrkbl = n + lwork_dgeqrf
                  wrkbl = MAX(wrkbl, n+lwork_dorgqr_n)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dgebrd)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dorgbr_q)
                  wrkbl = MAX(wrkbl, bdspac)
                  maxwrk = n*n + wrkbl
                  minwrk = MAX(3*n+m, bdspac)
                ELSE IF (wntus .AND. wntvo) THEN
                  wrkbl = n + lwork_dgeqrf
                  wrkbl = MAX(wrkbl, n+lwork_dorgqr_n)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dgebrd)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dorgbr_q)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dorgbr_p)
                  wrkbl = MAX(wrkbl, bdspac)
                  maxwrk = 2*n*n + wrkbl
                  minwrk = MAX(3*n+m, bdspac)
                ELSE IF (wntus .AND. wntvas) THEN
                  wrkbl = n + lwork_dgeqrf
                  wrkbl = MAX(wrkbl, n+lwork_dorgqr_n)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dgebrd)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dorgbr_q)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dorgbr_p)
                  wrkbl = MAX(wrkbl, bdspac)
                  maxwrk = n*n + wrkbl
                  minwrk = MAX(3*n+m, bdspac)
                ELSE IF (wntua .AND. wntvn) THEN
                  wrkbl = n + lwork_dgeqrf
                  wrkbl = MAX(wrkbl, n+lwork_dorgqr_m)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dgebrd)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dorgbr_q)
                  wrkbl = MAX(wrkbl, bdspac)
                  maxwrk = n*n + wrkbl
                  minwrk = MAX(3*n+m, bdspac)
                ELSE IF (wntua .AND. wntvo) THEN
                  wrkbl = n + lwork_dgeqrf
                  wrkbl = MAX(wrkbl, n+lwork_dorgqr_m)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dgebrd)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dorgbr_q)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dorgbr_p)
                  wrkbl = MAX(wrkbl, bdspac)
                  maxwrk = 2*n*n + wrkbl
                  minwrk = MAX(3*n+m, bdspac)
                ELSE IF (wntua .AND. wntvas) THEN
                  wrkbl = n + lwork_dgeqrf
                  wrkbl = MAX(wrkbl, n+lwork_dorgqr_m)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dgebrd)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dorgbr_q)
                  wrkbl = MAX(wrkbl, 3*n+lwork_dorgbr_p)
                  wrkbl = MAX(wrkbl, bdspac)
                  maxwrk = n*n + wrkbl
                  minwrk = MAX(3*n+m, bdspac)
                END IF
              ELSE
                CALL DGEBRD(m, n, a, lda, s, dum(1), dum(1), dum(1),        &
                  dum(1), -1_ip_, ierr)
                lwork_dgebrd = INT(dum(1))
                maxwrk = 3*n + lwork_dgebrd
                IF (wntus .OR. wntuo) THEN
                  CALL DORGBR('Q', m, n, n, a, lda, dum(1), dum(1), -1_ip_, &
                    ierr)
                  lwork_dorgbr_q = INT(dum(1))
                  maxwrk = MAX(maxwrk, 3*n+lwork_dorgbr_q)
                END IF
                IF (wntua) THEN
                  CALL DORGBR('Q', m, m, n, a, lda, dum(1), dum(1), -1_ip_, &
                    ierr)
                  lwork_dorgbr_q = INT(dum(1))
                  maxwrk = MAX(maxwrk, 3*n+lwork_dorgbr_q)
                END IF
                IF (.NOT. wntvn) THEN
                  maxwrk = MAX(maxwrk, 3*n+lwork_dorgbr_p)
                END IF
                maxwrk = MAX(maxwrk, bdspac)
                minwrk = MAX(3*n+m, bdspac)
              END IF
            ELSE IF (minmn>0) THEN
              mnthr = ILAENV2(6_ip_, 'GESVD', jobu//jobvt, m, n, 0_ip_,     &
                0_ip_)
              bdspac = 5*m
              CALL DGELQF(m, n, a, lda, dum(1), dum(1), -1_ip_, ierr)
              lwork_dgelqf = INT(dum(1))
              CALL DORGLQ(n, n, m, dum(1), n, dum(1), dum(1), -1_ip_, ierr)
              lwork_dorglq_n = INT(dum(1))
              CALL DORGLQ(m, n, m, a, lda, dum(1), dum(1), -1_ip_, ierr)
              lwork_dorglq_m = INT(dum(1))
              CALL DGEBRD(m, m, a, lda, s, dum(1), dum(1), dum(1), dum(1),  &
                -1_ip_, ierr)
              lwork_dgebrd = INT(dum(1))
              CALL DORGBR('P', m, m, m, a, n, dum(1), dum(1), -1_ip_, ierr)
              lwork_dorgbr_p = INT(dum(1))
              CALL DORGBR('Q', m, m, m, a, n, dum(1), dum(1), -1_ip_, ierr)
              lwork_dorgbr_q = INT(dum(1))
              IF (n>=mnthr) THEN
                IF (wntvn) THEN
                  maxwrk = m + lwork_dgelqf
                  maxwrk = MAX(maxwrk, 3*m+lwork_dgebrd)
                  IF (wntuo .OR. wntuas) maxwrk = MAX(maxwrk,               &
                    3*m+lwork_dorgbr_q)
                  maxwrk = MAX(maxwrk, bdspac)
                  minwrk = MAX(4*m, bdspac)
                ELSE IF (wntvo .AND. wntun) THEN
                  wrkbl = m + lwork_dgelqf
                  wrkbl = MAX(wrkbl, m+lwork_dorglq_m)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dgebrd)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dorgbr_p)
                  wrkbl = MAX(wrkbl, bdspac)
                  maxwrk = MAX(m*m+wrkbl, m*m+m*n+m)
                  minwrk = MAX(3*m+n, bdspac)
                ELSE IF (wntvo .AND. wntuas) THEN
                  wrkbl = m + lwork_dgelqf
                  wrkbl = MAX(wrkbl, m+lwork_dorglq_m)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dgebrd)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dorgbr_p)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dorgbr_q)
                  wrkbl = MAX(wrkbl, bdspac)
                  maxwrk = MAX(m*m+wrkbl, m*m+m*n+m)
                  minwrk = MAX(3*m+n, bdspac)
                ELSE IF (wntvs .AND. wntun) THEN
                  wrkbl = m + lwork_dgelqf
                  wrkbl = MAX(wrkbl, m+lwork_dorglq_m)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dgebrd)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dorgbr_p)
                  wrkbl = MAX(wrkbl, bdspac)
                  maxwrk = m*m + wrkbl
                  minwrk = MAX(3*m+n, bdspac)
                ELSE IF (wntvs .AND. wntuo) THEN
                  wrkbl = m + lwork_dgelqf
                  wrkbl = MAX(wrkbl, m+lwork_dorglq_m)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dgebrd)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dorgbr_p)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dorgbr_q)
                  wrkbl = MAX(wrkbl, bdspac)
                  maxwrk = 2*m*m + wrkbl
                  minwrk = MAX(3*m+n, bdspac)
                ELSE IF (wntvs .AND. wntuas) THEN
                  wrkbl = m + lwork_dgelqf
                  wrkbl = MAX(wrkbl, m+lwork_dorglq_m)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dgebrd)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dorgbr_p)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dorgbr_q)
                  wrkbl = MAX(wrkbl, bdspac)
                  maxwrk = m*m + wrkbl
                  minwrk = MAX(3*m+n, bdspac)
                ELSE IF (wntva .AND. wntun) THEN
                  wrkbl = m + lwork_dgelqf
                  wrkbl = MAX(wrkbl, m+lwork_dorglq_n)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dgebrd)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dorgbr_p)
                  wrkbl = MAX(wrkbl, bdspac)
                  maxwrk = m*m + wrkbl
                  minwrk = MAX(3*m+n, bdspac)
                ELSE IF (wntva .AND. wntuo) THEN
                  wrkbl = m + lwork_dgelqf
                  wrkbl = MAX(wrkbl, m+lwork_dorglq_n)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dgebrd)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dorgbr_p)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dorgbr_q)
                  wrkbl = MAX(wrkbl, bdspac)
                  maxwrk = 2*m*m + wrkbl
                  minwrk = MAX(3*m+n, bdspac)
                ELSE IF (wntva .AND. wntuas) THEN
                  wrkbl = m + lwork_dgelqf
                  wrkbl = MAX(wrkbl, m+lwork_dorglq_n)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dgebrd)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dorgbr_p)
                  wrkbl = MAX(wrkbl, 3*m+lwork_dorgbr_q)
                  wrkbl = MAX(wrkbl, bdspac)
                  maxwrk = m*m + wrkbl
                  minwrk = MAX(3*m+n, bdspac)
                END IF
              ELSE
                CALL DGEBRD(m, n, a, lda, s, dum(1), dum(1), dum(1),        &
                  dum(1), -1_ip_, ierr)
                lwork_dgebrd = INT(dum(1))
                maxwrk = 3*m + lwork_dgebrd
                IF (wntvs .OR. wntvo) THEN
                  CALL DORGBR('P', m, n, m, a, n, dum(1), dum(1), -1_ip_,   &
                    ierr)
                  lwork_dorgbr_p = INT(dum(1))
                  maxwrk = MAX(maxwrk, 3*m+lwork_dorgbr_p)
                END IF
                IF (wntva) THEN
                  CALL DORGBR('P', n, n, m, a, n, dum(1), dum(1), -1_ip_,   &
                    ierr)
                  lwork_dorgbr_p = INT(dum(1))
                  maxwrk = MAX(maxwrk, 3*m+lwork_dorgbr_p)
                END IF
                IF (.NOT. wntun) THEN
                  maxwrk = MAX(maxwrk, 3*m+lwork_dorgbr_q)
                END IF
                maxwrk = MAX(maxwrk, bdspac)
                minwrk = MAX(3*m+n, bdspac)
              END IF
            END IF
            maxwrk = MAX(maxwrk, minwrk)
            work(1) = REAL(maxwrk,rp_)
            IF (lwork<minwrk .AND. .NOT. lquery) THEN
              info = -13
            END IF
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GESVD', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          IF (m==0 .OR. n==0) THEN
            RETURN
          END IF
          eps = DLAMCH('P')
          smlnum = SQRT(DLAMCH('S'))/eps
          bignum = one/smlnum
          anrm = DLANGE('M', m, n, a, lda, dum)
          iscl = 0
          IF (anrm>zero .AND. anrm<smlnum) THEN
            iscl = 1
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, smlnum, m, n, a, lda,      &
              ierr)
          ELSE IF (anrm>bignum) THEN
            iscl = 1
            CALL DLASCL('G', 0_ip_, 0_ip_, anrm, bignum, m, n, a, lda,      &
              ierr)
          END IF
          IF (m>=n) THEN
            IF (m>=mnthr) THEN
              IF (wntun) THEN
                itau = 1
                iwork = itau + n
                CALL DGEQRF(m, n, a, lda, work(itau), work(iwork),          &
                  lwork-iwork+1, ierr)
                IF (n>1) THEN
                  CALL DLASET('L', n-1, n-1, zero, zero, a(2,1), lda)
                END IF
                ie = 1
                itauq = ie + n
                itaup = itauq + n
                iwork = itaup + n
                CALL DGEBRD(n, n, a, lda, s, work(ie), work(itauq),         &
                  work(itaup), work(iwork), lwork-iwork+1, ierr)
                ncvt = 0
                IF (wntvo .OR. wntvas) THEN
                  CALL DORGBR('P', n, n, n, a, lda, work(itaup),            &
                    work(iwork), lwork-iwork+1, ierr)
                  ncvt = n
                END IF
                iwork = ie + n
                CALL DBDSQR('U', n, ncvt, 0_ip_, 0_ip_, s, work(ie), a,     &
                  lda, dum, 1_ip_, dum, 1_ip_, work(iwork), info)
                IF (wntvas) CALL DLACPY('F', n, n, a, lda, vt, ldvt)
              ELSE IF (wntuo .AND. wntvn) THEN
                IF (lwork>=n*n+MAX(4*n,bdspac)) THEN
                  ir = 1
                  IF (lwork>=MAX(wrkbl,lda*n+n)+lda*n) THEN
                    ldwrku = lda
                    ldwrkr = lda
                  ELSE IF (lwork>=MAX(wrkbl,lda*n+n)+n*n) THEN
                    ldwrku = lda
                    ldwrkr = n
                  ELSE
                    ldwrku = (lwork-n*n-n)/n
                    ldwrkr = n
                  END IF
                  itau = ir + ldwrkr*n
                  iwork = itau + n
                  CALL DGEQRF(m, n, a, lda, work(itau), work(iwork),        &
                    lwork-iwork+1, ierr)
                  CALL DLACPY('U', n, n, a, lda, work(ir), ldwrkr)
                  CALL DLASET('L', n-1, n-1, zero, zero, work(ir+1),        &
                    ldwrkr)
                  CALL DORGQR(m, n, n, a, lda, work(itau), work(iwork),     &
                    lwork-iwork+1, ierr)
                  ie = itau
                  itauq = ie + n
                  itaup = itauq + n
                  iwork = itaup + n
                  CALL DGEBRD(n, n, work(ir), ldwrkr, s, work(ie),          &
                    work(itauq), work(itaup), work(iwork), lwork-iwork+1,   &
                    ierr)
                  CALL DORGBR('Q', n, n, n, work(ir), ldwrkr, work(itauq),  &
                    work(iwork), lwork-iwork+1, ierr)
                  iwork = ie + n
                  CALL DBDSQR('U', n, 0_ip_, n, 0_ip_, s, work(ie), dum,    &
                    1_ip_, work(ir), ldwrkr, dum, 1_ip_, work(iwork), info)
                  iu = ie + n
                  DO i = 1, m, ldwrku
                    chunk = MIN(m-i+1, ldwrku)
                    CALL DGEMM('N', 'N', chunk, n, n, one, a(i,1), lda,     &
                      work(ir), ldwrkr, zero, work(iu), ldwrku)
                    CALL DLACPY('F', chunk, n, work(iu), ldwrku, a(i,1),    &
                      lda)
                  END DO
                ELSE
                  ie = 1
                  itauq = ie + n
                  itaup = itauq + n
                  iwork = itaup + n
                  CALL DGEBRD(m, n, a, lda, s, work(ie), work(itauq),       &
                    work(itaup), work(iwork), lwork-iwork+1, ierr)
                  CALL DORGBR('Q', m, n, n, a, lda, work(itauq),            &
                    work(iwork), lwork-iwork+1, ierr)
                  iwork = ie + n
                  CALL DBDSQR('U', n, 0_ip_, m, 0_ip_, s, work(ie), dum,    &
                    1_ip_, a, lda, dum, 1_ip_, work(iwork), info)
                END IF
              ELSE IF (wntuo .AND. wntvas) THEN
                IF (lwork>=n*n+MAX(4*n,bdspac)) THEN
                  ir = 1
                  IF (lwork>=MAX(wrkbl,lda*n+n)+lda*n) THEN
                    ldwrku = lda
                    ldwrkr = lda
                  ELSE IF (lwork>=MAX(wrkbl,lda*n+n)+n*n) THEN
                    ldwrku = lda
                    ldwrkr = n
                  ELSE
                    ldwrku = (lwork-n*n-n)/n
                    ldwrkr = n
                  END IF
                  itau = ir + ldwrkr*n
                  iwork = itau + n
                  CALL DGEQRF(m, n, a, lda, work(itau), work(iwork),        &
                    lwork-iwork+1, ierr)
                  CALL DLACPY('U', n, n, a, lda, vt, ldvt)
                  IF (n>1) CALL DLASET('L', n-1, n-1, zero, zero, vt(2,1),  &
                    ldvt)
                  CALL DORGQR(m, n, n, a, lda, work(itau), work(iwork),     &
                    lwork-iwork+1, ierr)
                  ie = itau
                  itauq = ie + n
                  itaup = itauq + n
                  iwork = itaup + n
                  CALL DGEBRD(n, n, vt, ldvt, s, work(ie), work(itauq),     &
                    work(itaup), work(iwork), lwork-iwork+1, ierr)
                  CALL DLACPY('L', n, n, vt, ldvt, work(ir), ldwrkr)
                  CALL DORGBR('Q', n, n, n, work(ir), ldwrkr, work(itauq),  &
                    work(iwork), lwork-iwork+1, ierr)
                  CALL DORGBR('P', n, n, n, vt, ldvt, work(itaup),          &
                    work(iwork), lwork-iwork+1, ierr)
                  iwork = ie + n
                  CALL DBDSQR('U', n, n, n, 0_ip_, s, work(ie), vt, ldvt,   &
                    work(ir), ldwrkr, dum, 1_ip_, work(iwork), info)
                  iu = ie + n
                  DO i = 1, m, ldwrku
                    chunk = MIN(m-i+1, ldwrku)
                    CALL DGEMM('N', 'N', chunk, n, n, one, a(i,1), lda,     &
                      work(ir), ldwrkr, zero, work(iu), ldwrku)
                    CALL DLACPY('F', chunk, n, work(iu), ldwrku, a(i,1),    &
                      lda)
                  END DO
                ELSE
                  itau = 1
                  iwork = itau + n
                  CALL DGEQRF(m, n, a, lda, work(itau), work(iwork),        &
                    lwork-iwork+1, ierr)
                  CALL DLACPY('U', n, n, a, lda, vt, ldvt)
                  IF (n>1) CALL DLASET('L', n-1, n-1, zero, zero, vt(2,1),  &
                    ldvt)
                  CALL DORGQR(m, n, n, a, lda, work(itau), work(iwork),     &
                    lwork-iwork+1, ierr)
                  ie = itau
                  itauq = ie + n
                  itaup = itauq + n
                  iwork = itaup + n
                  CALL DGEBRD(n, n, vt, ldvt, s, work(ie), work(itauq),     &
                    work(itaup), work(iwork), lwork-iwork+1, ierr)
                  CALL DORMBR('Q', 'R', 'N', m, n, n, vt, ldvt,             &
                    work(itauq), a, lda, work(iwork), lwork-iwork+1, ierr)
                  CALL DORGBR('P', n, n, n, vt, ldvt, work(itaup),          &
                    work(iwork), lwork-iwork+1, ierr)
                  iwork = ie + n
                  CALL DBDSQR('U', n, n, m, 0_ip_, s, work(ie), vt, ldvt,   &
                    a, lda, dum, 1_ip_, work(iwork), info)
                END IF
              ELSE IF (wntus) THEN
                IF (wntvn) THEN
                  IF (lwork>=n*n+MAX(4*n,bdspac)) THEN
                    ir = 1
                    IF (lwork>=wrkbl+lda*n) THEN
                      ldwrkr = lda
                    ELSE
                      ldwrkr = n
                    END IF
                    itau = ir + ldwrkr*n
                    iwork = itau + n
                    CALL DGEQRF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('U', n, n, a, lda, work(ir), ldwrkr)
                    CALL DLASET('L', n-1, n-1, zero, zero, work(ir+1),      &
                      ldwrkr)
                    CALL DORGQR(m, n, n, a, lda, work(itau), work(iwork),   &
                      lwork-iwork+1, ierr)
                    ie = itau
                    itauq = ie + n
                    itaup = itauq + n
                    iwork = itaup + n
                    CALL DGEBRD(n, n, work(ir), ldwrkr, s, work(ie),        &
                      work(itauq), work(itaup), work(iwork), lwork-iwork+1, &
                      ierr)
                    CALL DORGBR('Q', n, n, n, work(ir), ldwrkr,             &
                      work(itauq), work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + n
                    CALL DBDSQR('U', n, 0_ip_, n, 0_ip_, s, work(ie), dum,  &
                      1_ip_, work(ir), ldwrkr, dum, 1_ip_, work(iwork), info)
                    CALL DGEMM('N', 'N', m, n, n, one, a, lda, work(ir),    &
                      ldwrkr, zero, u, ldu)
                  ELSE
                    itau = 1
                    iwork = itau + n
                    CALL DGEQRF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('L', m, n, a, lda, u, ldu)
                    CALL DORGQR(m, n, n, u, ldu, work(itau), work(iwork),   &
                      lwork-iwork+1, ierr)
                    ie = itau
                    itauq = ie + n
                    itaup = itauq + n
                    iwork = itaup + n
                    IF (n>1) THEN
                      CALL DLASET('L', n-1, n-1, zero, zero, a(2,1), lda)
                    END IF
                    CALL DGEBRD(n, n, a, lda, s, work(ie), work(itauq),     &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    CALL DORMBR('Q', 'R', 'N', m, n, n, a, lda,             &
                      work(itauq), u, ldu, work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + n
                    CALL DBDSQR('U', n, 0_ip_, m, 0_ip_, s, work(ie), dum,  &
                      1_ip_, u, ldu, dum, 1_ip_, work(iwork), info)
                  END IF
                ELSE IF (wntvo) THEN
                  IF (lwork>=2*n*n+MAX(4*n,bdspac)) THEN
                    iu = 1
                    IF (lwork>=wrkbl+2*lda*n) THEN
                      ldwrku = lda
                      ir = iu + ldwrku*n
                      ldwrkr = lda
                    ELSE IF (lwork>=wrkbl+(lda+n)*n) THEN
                      ldwrku = lda
                      ir = iu + ldwrku*n
                      ldwrkr = n
                    ELSE
                      ldwrku = n
                      ir = iu + ldwrku*n
                      ldwrkr = n
                    END IF
                    itau = ir + ldwrkr*n
                    iwork = itau + n
                    CALL DGEQRF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('U', n, n, a, lda, work(iu), ldwrku)
                    CALL DLASET('L', n-1, n-1, zero, zero, work(iu+1),      &
                      ldwrku)
                    CALL DORGQR(m, n, n, a, lda, work(itau), work(iwork),   &
                      lwork-iwork+1, ierr)
                    ie = itau
                    itauq = ie + n
                    itaup = itauq + n
                    iwork = itaup + n
                    CALL DGEBRD(n, n, work(iu), ldwrku, s, work(ie),        &
                      work(itauq), work(itaup), work(iwork), lwork-iwork+1, &
                      ierr)
                    CALL DLACPY('U', n, n, work(iu), ldwrku, work(ir),      &
                      ldwrkr)
                    CALL DORGBR('Q', n, n, n, work(iu), ldwrku,             &
                      work(itauq), work(iwork), lwork-iwork+1, ierr)
                    CALL DORGBR('P', n, n, n, work(ir), ldwrkr,             &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + n
                    CALL DBDSQR('U', n, n, n, 0_ip_, s, work(ie), work(ir), &
                      ldwrkr, work(iu), ldwrku, dum, 1_ip_, work(iwork),    &
                      info)
                    CALL DGEMM('N', 'N', m, n, n, one, a, lda, work(iu),    &
                      ldwrku, zero, u, ldu)
                    CALL DLACPY('F', n, n, work(ir), ldwrkr, a, lda)
                  ELSE
                    itau = 1
                    iwork = itau + n
                    CALL DGEQRF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('L', m, n, a, lda, u, ldu)
                    CALL DORGQR(m, n, n, u, ldu, work(itau), work(iwork),   &
                      lwork-iwork+1, ierr)
                    ie = itau
                    itauq = ie + n
                    itaup = itauq + n
                    iwork = itaup + n
                    IF (n>1) THEN
                      CALL DLASET('L', n-1, n-1, zero, zero, a(2,1), lda)
                    END IF
                    CALL DGEBRD(n, n, a, lda, s, work(ie), work(itauq),     &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    CALL DORMBR('Q', 'R', 'N', m, n, n, a, lda,             &
                      work(itauq), u, ldu, work(iwork), lwork-iwork+1, ierr)
                    CALL DORGBR('P', n, n, n, a, lda, work(itaup),          &
                      work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + n
                    CALL DBDSQR('U', n, n, m, 0_ip_, s, work(ie), a, lda,   &
                      u, ldu, dum, 1_ip_, work(iwork), info)
                  END IF
                ELSE IF (wntvas) THEN
                  IF (lwork>=n*n+MAX(4*n,bdspac)) THEN
                    iu = 1
                    IF (lwork>=wrkbl+lda*n) THEN
                      ldwrku = lda
                    ELSE
                      ldwrku = n
                    END IF
                    itau = iu + ldwrku*n
                    iwork = itau + n
                    CALL DGEQRF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('U', n, n, a, lda, work(iu), ldwrku)
                    CALL DLASET('L', n-1, n-1, zero, zero, work(iu+1),      &
                      ldwrku)
                    CALL DORGQR(m, n, n, a, lda, work(itau), work(iwork),   &
                      lwork-iwork+1, ierr)
                    ie = itau
                    itauq = ie + n
                    itaup = itauq + n
                    iwork = itaup + n
                    CALL DGEBRD(n, n, work(iu), ldwrku, s, work(ie),        &
                      work(itauq), work(itaup), work(iwork), lwork-iwork+1, &
                      ierr)
                    CALL DLACPY('U', n, n, work(iu), ldwrku, vt, ldvt)
                    CALL DORGBR('Q', n, n, n, work(iu), ldwrku,             &
                      work(itauq), work(iwork), lwork-iwork+1, ierr)
                    CALL DORGBR('P', n, n, n, vt, ldvt, work(itaup),        &
                      work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + n
                    CALL DBDSQR('U', n, n, n, 0_ip_, s, work(ie), vt, ldvt, &
                      work(iu), ldwrku, dum, 1_ip_, work(iwork), info)
                    CALL DGEMM('N', 'N', m, n, n, one, a, lda, work(iu),    &
                      ldwrku, zero, u, ldu)
                  ELSE
                    itau = 1
                    iwork = itau + n
                    CALL DGEQRF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('L', m, n, a, lda, u, ldu)
                    CALL DORGQR(m, n, n, u, ldu, work(itau), work(iwork),   &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('U', n, n, a, lda, vt, ldvt)
                    IF (n>1) CALL DLASET('L', n-1, n-1, zero, zero, vt(2,   &
                      1), ldvt)
                    ie = itau
                    itauq = ie + n
                    itaup = itauq + n
                    iwork = itaup + n
                    CALL DGEBRD(n, n, vt, ldvt, s, work(ie), work(itauq),   &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    CALL DORMBR('Q', 'R', 'N', m, n, n, vt, ldvt,           &
                      work(itauq), u, ldu, work(iwork), lwork-iwork+1, ierr)
                    CALL DORGBR('P', n, n, n, vt, ldvt, work(itaup),        &
                      work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + n
                    CALL DBDSQR('U', n, n, m, 0_ip_, s, work(ie), vt, ldvt, &
                      u, ldu, dum, 1_ip_, work(iwork), info)
                  END IF
                END IF
              ELSE IF (wntua) THEN
                IF (wntvn) THEN
                  IF (lwork>=n*n+MAX(n+m,4*n,bdspac)) THEN
                    ir = 1
                    IF (lwork>=wrkbl+lda*n) THEN
                      ldwrkr = lda
                    ELSE
                      ldwrkr = n
                    END IF
                    itau = ir + ldwrkr*n
                    iwork = itau + n
                    CALL DGEQRF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('L', m, n, a, lda, u, ldu)
                    CALL DLACPY('U', n, n, a, lda, work(ir), ldwrkr)
                    CALL DLASET('L', n-1, n-1, zero, zero, work(ir+1),      &
                      ldwrkr)
                    CALL DORGQR(m, m, n, u, ldu, work(itau), work(iwork),   &
                      lwork-iwork+1, ierr)
                    ie = itau
                    itauq = ie + n
                    itaup = itauq + n
                    iwork = itaup + n
                    CALL DGEBRD(n, n, work(ir), ldwrkr, s, work(ie),        &
                      work(itauq), work(itaup), work(iwork), lwork-iwork+1, &
                      ierr)
                    CALL DORGBR('Q', n, n, n, work(ir), ldwrkr,             &
                      work(itauq), work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + n
                    CALL DBDSQR('U', n, 0_ip_, n, 0_ip_, s, work(ie), dum,  &
                      1_ip_, work(ir), ldwrkr, dum, 1_ip_, work(iwork), info)
                    CALL DGEMM('N', 'N', m, n, n, one, u, ldu, work(ir),    &
                      ldwrkr, zero, a, lda)
                    CALL DLACPY('F', m, n, a, lda, u, ldu)
                  ELSE
                    itau = 1
                    iwork = itau + n
                    CALL DGEQRF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('L', m, n, a, lda, u, ldu)
                    CALL DORGQR(m, m, n, u, ldu, work(itau), work(iwork),   &
                      lwork-iwork+1, ierr)
                    ie = itau
                    itauq = ie + n
                    itaup = itauq + n
                    iwork = itaup + n
                    IF (n>1) THEN
                      CALL DLASET('L', n-1, n-1, zero, zero, a(2,1), lda)
                    END IF
                    CALL DGEBRD(n, n, a, lda, s, work(ie), work(itauq),     &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    CALL DORMBR('Q', 'R', 'N', m, n, n, a, lda,             &
                      work(itauq), u, ldu, work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + n
                    CALL DBDSQR('U', n, 0_ip_, m, 0_ip_, s, work(ie), dum,  &
                      1_ip_, u, ldu, dum, 1_ip_, work(iwork), info)
                  END IF
                ELSE IF (wntvo) THEN
                  IF (lwork>=2*n*n+MAX(n+m,4*n,bdspac)) THEN
                    iu = 1
                    IF (lwork>=wrkbl+2*lda*n) THEN
                      ldwrku = lda
                      ir = iu + ldwrku*n
                      ldwrkr = lda
                    ELSE IF (lwork>=wrkbl+(lda+n)*n) THEN
                      ldwrku = lda
                      ir = iu + ldwrku*n
                      ldwrkr = n
                    ELSE
                      ldwrku = n
                      ir = iu + ldwrku*n
                      ldwrkr = n
                    END IF
                    itau = ir + ldwrkr*n
                    iwork = itau + n
                    CALL DGEQRF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('L', m, n, a, lda, u, ldu)
                    CALL DORGQR(m, m, n, u, ldu, work(itau), work(iwork),   &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('U', n, n, a, lda, work(iu), ldwrku)
                    CALL DLASET('L', n-1, n-1, zero, zero, work(iu+1),      &
                      ldwrku)
                    ie = itau
                    itauq = ie + n
                    itaup = itauq + n
                    iwork = itaup + n
                    CALL DGEBRD(n, n, work(iu), ldwrku, s, work(ie),        &
                      work(itauq), work(itaup), work(iwork), lwork-iwork+1, &
                      ierr)
                    CALL DLACPY('U', n, n, work(iu), ldwrku, work(ir),      &
                      ldwrkr)
                    CALL DORGBR('Q', n, n, n, work(iu), ldwrku,             &
                      work(itauq), work(iwork), lwork-iwork+1, ierr)
                    CALL DORGBR('P', n, n, n, work(ir), ldwrkr,             &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + n
                    CALL DBDSQR('U', n, n, n, 0_ip_, s, work(ie), work(ir), &
                      ldwrkr, work(iu), ldwrku, dum, 1_ip_, work(iwork),    &
                      info)
                    CALL DGEMM('N', 'N', m, n, n, one, u, ldu, work(iu),    &
                      ldwrku, zero, a, lda)
                    CALL DLACPY('F', m, n, a, lda, u, ldu)
                    CALL DLACPY('F', n, n, work(ir), ldwrkr, a, lda)
                  ELSE
                    itau = 1
                    iwork = itau + n
                    CALL DGEQRF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('L', m, n, a, lda, u, ldu)
                    CALL DORGQR(m, m, n, u, ldu, work(itau), work(iwork),   &
                      lwork-iwork+1, ierr)
                    ie = itau
                    itauq = ie + n
                    itaup = itauq + n
                    iwork = itaup + n
                    IF (n>1) THEN
                      CALL DLASET('L', n-1, n-1, zero, zero, a(2,1), lda)
                    END IF
                    CALL DGEBRD(n, n, a, lda, s, work(ie), work(itauq),     &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    CALL DORMBR('Q', 'R', 'N', m, n, n, a, lda,             &
                      work(itauq), u, ldu, work(iwork), lwork-iwork+1, ierr)
                    CALL DORGBR('P', n, n, n, a, lda, work(itaup),          &
                      work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + n
                    CALL DBDSQR('U', n, n, m, 0_ip_, s, work(ie), a, lda,   &
                      u, ldu, dum, 1_ip_, work(iwork), info)
                  END IF
                ELSE IF (wntvas) THEN
                  IF (lwork>=n*n+MAX(n+m,4*n,bdspac)) THEN
                    iu = 1
                    IF (lwork>=wrkbl+lda*n) THEN
                      ldwrku = lda
                    ELSE
                      ldwrku = n
                    END IF
                    itau = iu + ldwrku*n
                    iwork = itau + n
                    CALL DGEQRF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('L', m, n, a, lda, u, ldu)
                    CALL DORGQR(m, m, n, u, ldu, work(itau), work(iwork),   &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('U', n, n, a, lda, work(iu), ldwrku)
                    CALL DLASET('L', n-1, n-1, zero, zero, work(iu+1),      &
                      ldwrku)
                    ie = itau
                    itauq = ie + n
                    itaup = itauq + n
                    iwork = itaup + n
                    CALL DGEBRD(n, n, work(iu), ldwrku, s, work(ie),        &
                      work(itauq), work(itaup), work(iwork), lwork-iwork+1, &
                      ierr)
                    CALL DLACPY('U', n, n, work(iu), ldwrku, vt, ldvt)
                    CALL DORGBR('Q', n, n, n, work(iu), ldwrku,             &
                      work(itauq), work(iwork), lwork-iwork+1, ierr)
                    CALL DORGBR('P', n, n, n, vt, ldvt, work(itaup),        &
                      work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + n
                    CALL DBDSQR('U', n, n, n, 0_ip_, s, work(ie), vt, ldvt, &
                      work(iu), ldwrku, dum, 1_ip_, work(iwork), info)
                    CALL DGEMM('N', 'N', m, n, n, one, u, ldu, work(iu),    &
                      ldwrku, zero, a, lda)
                    CALL DLACPY('F', m, n, a, lda, u, ldu)
                  ELSE
                    itau = 1
                    iwork = itau + n
                    CALL DGEQRF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('L', m, n, a, lda, u, ldu)
                    CALL DORGQR(m, m, n, u, ldu, work(itau), work(iwork),   &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('U', n, n, a, lda, vt, ldvt)
                    IF (n>1) CALL DLASET('L', n-1, n-1, zero, zero, vt(2,   &
                      1), ldvt)
                    ie = itau
                    itauq = ie + n
                    itaup = itauq + n
                    iwork = itaup + n
                    CALL DGEBRD(n, n, vt, ldvt, s, work(ie), work(itauq),   &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    CALL DORMBR('Q', 'R', 'N', m, n, n, vt, ldvt,           &
                      work(itauq), u, ldu, work(iwork), lwork-iwork+1, ierr)
                    CALL DORGBR('P', n, n, n, vt, ldvt, work(itaup),        &
                      work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + n
                    CALL DBDSQR('U', n, n, m, 0_ip_, s, work(ie), vt, ldvt, &
                      u, ldu, dum, 1_ip_, work(iwork), info)
                  END IF
                END IF
              END IF
            ELSE
              ie = 1
              itauq = ie + n
              itaup = itauq + n
              iwork = itaup + n
              CALL DGEBRD(m, n, a, lda, s, work(ie), work(itauq),           &
                work(itaup), work(iwork), lwork-iwork+1, ierr)
              IF (wntuas) THEN
                CALL DLACPY('L', m, n, a, lda, u, ldu)
                IF (wntus) ncu = n
                IF (wntua) ncu = m
                CALL DORGBR('Q', m, ncu, n, u, ldu, work(itauq),            &
                  work(iwork), lwork-iwork+1, ierr)
              END IF
              IF (wntvas) THEN
                CALL DLACPY('U', n, n, a, lda, vt, ldvt)
                CALL DORGBR('P', n, n, n, vt, ldvt, work(itaup),            &
                  work(iwork), lwork-iwork+1, ierr)
              END IF
              IF (wntuo) THEN
                CALL DORGBR('Q', m, n, n, a, lda, work(itauq), work(iwork), &
                  lwork-iwork+1, ierr)
              END IF
              IF (wntvo) THEN
                CALL DORGBR('P', n, n, n, a, lda, work(itaup), work(iwork), &
                  lwork-iwork+1, ierr)
              END IF
              iwork = ie + n
              IF (wntuas .OR. wntuo) nru = m
              IF (wntun) nru = 0
              IF (wntvas .OR. wntvo) ncvt = n
              IF (wntvn) ncvt = 0
              IF ((.NOT. wntuo) .AND. (.NOT. wntvo)) THEN
                CALL DBDSQR('U', n, ncvt, nru, 0_ip_, s, work(ie), vt,      &
                  ldvt, u, ldu, dum, 1_ip_, work(iwork), info)
              ELSE IF ((.NOT. wntuo) .AND. wntvo) THEN
                CALL DBDSQR('U', n, ncvt, nru, 0_ip_, s, work(ie), a, lda,  &
                  u, ldu, dum, 1_ip_, work(iwork), info)
              ELSE
                CALL DBDSQR('U', n, ncvt, nru, 0_ip_, s, work(ie), vt,      &
                  ldvt, a, lda, dum, 1_ip_, work(iwork), info)
              END IF
            END IF
          ELSE
            IF (n>=mnthr) THEN
              IF (wntvn) THEN
                itau = 1
                iwork = itau + m
                CALL DGELQF(m, n, a, lda, work(itau), work(iwork),          &
                  lwork-iwork+1, ierr)
                CALL DLASET('U', m-1, m-1, zero, zero, a(1,2), lda)
                ie = 1
                itauq = ie + m
                itaup = itauq + m
                iwork = itaup + m
                CALL DGEBRD(m, m, a, lda, s, work(ie), work(itauq),         &
                  work(itaup), work(iwork), lwork-iwork+1, ierr)
                IF (wntuo .OR. wntuas) THEN
                  CALL DORGBR('Q', m, m, m, a, lda, work(itauq),            &
                    work(iwork), lwork-iwork+1, ierr)
                END IF
                iwork = ie + m
                nru = 0
                IF (wntuo .OR. wntuas) nru = m
                CALL DBDSQR('U', m, 0_ip_, nru, 0_ip_, s, work(ie), dum,    &
                  1_ip_, a, lda, dum, 1_ip_, work(iwork), info)
                IF (wntuas) CALL DLACPY('F', m, m, a, lda, u, ldu)
              ELSE IF (wntvo .AND. wntun) THEN
                IF (lwork>=m*m+MAX(4*m,bdspac)) THEN
                  ir = 1
                  IF (lwork>=MAX(wrkbl,lda*n+m)+lda*m) THEN
                    ldwrku = lda
                    chunk = n
                    ldwrkr = lda
                  ELSE IF (lwork>=MAX(wrkbl,lda*n+m)+m*m) THEN
                    ldwrku = lda
                    chunk = n
                    ldwrkr = m
                  ELSE
                    ldwrku = m
                    chunk = (lwork-m*m-m)/m
                    ldwrkr = m
                  END IF
                  itau = ir + ldwrkr*m
                  iwork = itau + m
                  CALL DGELQF(m, n, a, lda, work(itau), work(iwork),        &
                    lwork-iwork+1, ierr)
                  CALL DLACPY('L', m, m, a, lda, work(ir), ldwrkr)
                  CALL DLASET('U', m-1, m-1, zero, zero, work(ir+ldwrkr),   &
                    ldwrkr)
                  CALL DORGLQ(m, n, m, a, lda, work(itau), work(iwork),     &
                    lwork-iwork+1, ierr)
                  ie = itau
                  itauq = ie + m
                  itaup = itauq + m
                  iwork = itaup + m
                  CALL DGEBRD(m, m, work(ir), ldwrkr, s, work(ie),          &
                    work(itauq), work(itaup), work(iwork), lwork-iwork+1,   &
                    ierr)
                  CALL DORGBR('P', m, m, m, work(ir), ldwrkr, work(itaup),  &
                    work(iwork), lwork-iwork+1, ierr)
                  iwork = ie + m
                  CALL DBDSQR('U', m, m, 0_ip_, 0_ip_, s, work(ie),         &
                    work(ir), ldwrkr, dum, 1_ip_, dum, 1_ip_, work(iwork),  &
                    info)
                  iu = ie + m
                  DO i = 1, n, chunk
                    blk = MIN(n-i+1, chunk)
                    CALL DGEMM('N', 'N', m, blk, m, one, work(ir), ldwrkr,  &
                      a(1,i), lda, zero, work(iu), ldwrku)
                    CALL DLACPY('F', m, blk, work(iu), ldwrku, a(1,i), lda)
                  END DO
                ELSE
                  ie = 1
                  itauq = ie + m
                  itaup = itauq + m
                  iwork = itaup + m
                  CALL DGEBRD(m, n, a, lda, s, work(ie), work(itauq),       &
                    work(itaup), work(iwork), lwork-iwork+1, ierr)
                  CALL DORGBR('P', m, n, m, a, lda, work(itaup),            &
                    work(iwork), lwork-iwork+1, ierr)
                  iwork = ie + m
                  CALL DBDSQR('L', m, n, 0_ip_, 0_ip_, s, work(ie), a, lda, &
                    dum, 1_ip_, dum, 1_ip_, work(iwork), info)
                END IF
              ELSE IF (wntvo .AND. wntuas) THEN
                IF (lwork>=m*m+MAX(4*m,bdspac)) THEN
                  ir = 1
                  IF (lwork>=MAX(wrkbl,lda*n+m)+lda*m) THEN
                    ldwrku = lda
                    chunk = n
                    ldwrkr = lda
                  ELSE IF (lwork>=MAX(wrkbl,lda*n+m)+m*m) THEN
                    ldwrku = lda
                    chunk = n
                    ldwrkr = m
                  ELSE
                    ldwrku = m
                    chunk = (lwork-m*m-m)/m
                    ldwrkr = m
                  END IF
                  itau = ir + ldwrkr*m
                  iwork = itau + m
                  CALL DGELQF(m, n, a, lda, work(itau), work(iwork),        &
                    lwork-iwork+1, ierr)
                  CALL DLACPY('L', m, m, a, lda, u, ldu)
                  CALL DLASET('U', m-1, m-1, zero, zero, u(1,2), ldu)
                  CALL DORGLQ(m, n, m, a, lda, work(itau), work(iwork),     &
                    lwork-iwork+1, ierr)
                  ie = itau
                  itauq = ie + m
                  itaup = itauq + m
                  iwork = itaup + m
                  CALL DGEBRD(m, m, u, ldu, s, work(ie), work(itauq),       &
                    work(itaup), work(iwork), lwork-iwork+1, ierr)
                  CALL DLACPY('U', m, m, u, ldu, work(ir), ldwrkr)
                  CALL DORGBR('P', m, m, m, work(ir), ldwrkr, work(itaup),  &
                    work(iwork), lwork-iwork+1, ierr)
                  CALL DORGBR('Q', m, m, m, u, ldu, work(itauq),            &
                    work(iwork), lwork-iwork+1, ierr)
                  iwork = ie + m
                  CALL DBDSQR('U', m, m, m, 0_ip_, s, work(ie), work(ir),   &
                    ldwrkr, u, ldu, dum, 1_ip_, work(iwork), info)
                  iu = ie + m
                  DO i = 1, n, chunk
                    blk = MIN(n-i+1, chunk)
                    CALL DGEMM('N', 'N', m, blk, m, one, work(ir), ldwrkr,  &
                      a(1,i), lda, zero, work(iu), ldwrku)
                    CALL DLACPY('F', m, blk, work(iu), ldwrku, a(1,i), lda)
                  END DO
                ELSE
                  itau = 1
                  iwork = itau + m
                  CALL DGELQF(m, n, a, lda, work(itau), work(iwork),        &
                    lwork-iwork+1, ierr)
                  CALL DLACPY('L', m, m, a, lda, u, ldu)
                  CALL DLASET('U', m-1, m-1, zero, zero, u(1,2), ldu)
                  CALL DORGLQ(m, n, m, a, lda, work(itau), work(iwork),     &
                    lwork-iwork+1, ierr)
                  ie = itau
                  itauq = ie + m
                  itaup = itauq + m
                  iwork = itaup + m
                  CALL DGEBRD(m, m, u, ldu, s, work(ie), work(itauq),       &
                    work(itaup), work(iwork), lwork-iwork+1, ierr)
                  CALL DORMBR('P', 'L', 'T', m, n, m, u, ldu, work(itaup),  &
                    a, lda, work(iwork), lwork-iwork+1, ierr)
                  CALL DORGBR('Q', m, m, m, u, ldu, work(itauq),            &
                    work(iwork), lwork-iwork+1, ierr)
                  iwork = ie + m
                  CALL DBDSQR('U', m, n, m, 0_ip_, s, work(ie), a, lda, u,  &
                    ldu, dum, 1_ip_, work(iwork), info)
                END IF
              ELSE IF (wntvs) THEN
                IF (wntun) THEN
                  IF (lwork>=m*m+MAX(4*m,bdspac)) THEN
                    ir = 1
                    IF (lwork>=wrkbl+lda*m) THEN
                      ldwrkr = lda
                    ELSE
                      ldwrkr = m
                    END IF
                    itau = ir + ldwrkr*m
                    iwork = itau + m
                    CALL DGELQF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('L', m, m, a, lda, work(ir), ldwrkr)
                    CALL DLASET('U', m-1, m-1, zero, zero, work(ir+ldwrkr), &
                      ldwrkr)
                    CALL DORGLQ(m, n, m, a, lda, work(itau), work(iwork),   &
                      lwork-iwork+1, ierr)
                    ie = itau
                    itauq = ie + m
                    itaup = itauq + m
                    iwork = itaup + m
                    CALL DGEBRD(m, m, work(ir), ldwrkr, s, work(ie),        &
                      work(itauq), work(itaup), work(iwork), lwork-iwork+1, &
                      ierr)
                    CALL DORGBR('P', m, m, m, work(ir), ldwrkr,             &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + m
                    CALL DBDSQR('U', m, m, 0_ip_, 0_ip_, s, work(ie),       &
                      work(ir), ldwrkr, dum, 1_ip_, dum, 1_ip_, work(iwork),&
                      info)
                    CALL DGEMM('N', 'N', m, n, m, one, work(ir), ldwrkr, a, &
                      lda, zero, vt, ldvt)
                  ELSE
                    itau = 1
                    iwork = itau + m
                    CALL DGELQF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('U', m, n, a, lda, vt, ldvt)
                    CALL DORGLQ(m, n, m, vt, ldvt, work(itau), work(iwork), &
                      lwork-iwork+1, ierr)
                    ie = itau
                    itauq = ie + m
                    itaup = itauq + m
                    iwork = itaup + m
                    CALL DLASET('U', m-1, m-1, zero, zero, a(1,2), lda)
                    CALL DGEBRD(m, m, a, lda, s, work(ie), work(itauq),     &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    CALL DORMBR('P', 'L', 'T', m, n, m, a, lda,             &
                      work(itaup), vt, ldvt, work(iwork), lwork-iwork+1,    &
                      ierr)
                    iwork = ie + m
                    CALL DBDSQR('U', m, n, 0_ip_, 0_ip_, s, work(ie), vt,   &
                      ldvt, dum, 1_ip_, dum, 1_ip_, work(iwork), info)
                  END IF
                ELSE IF (wntuo) THEN
                  IF (lwork>=2*m*m+MAX(4*m,bdspac)) THEN
                    iu = 1
                    IF (lwork>=wrkbl+2*lda*m) THEN
                      ldwrku = lda
                      ir = iu + ldwrku*m
                      ldwrkr = lda
                    ELSE IF (lwork>=wrkbl+(lda+m)*m) THEN
                      ldwrku = lda
                      ir = iu + ldwrku*m
                      ldwrkr = m
                    ELSE
                      ldwrku = m
                      ir = iu + ldwrku*m
                      ldwrkr = m
                    END IF
                    itau = ir + ldwrkr*m
                    iwork = itau + m
                    CALL DGELQF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('L', m, m, a, lda, work(iu), ldwrku)
                    CALL DLASET('U', m-1, m-1, zero, zero, work(iu+ldwrku), &
                      ldwrku)
                    CALL DORGLQ(m, n, m, a, lda, work(itau), work(iwork),   &
                      lwork-iwork+1, ierr)
                    ie = itau
                    itauq = ie + m
                    itaup = itauq + m
                    iwork = itaup + m
                    CALL DGEBRD(m, m, work(iu), ldwrku, s, work(ie),        &
                      work(itauq), work(itaup), work(iwork), lwork-iwork+1, &
                      ierr)
                    CALL DLACPY('L', m, m, work(iu), ldwrku, work(ir),      &
                      ldwrkr)
                    CALL DORGBR('P', m, m, m, work(iu), ldwrku,             &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    CALL DORGBR('Q', m, m, m, work(ir), ldwrkr,             &
                      work(itauq), work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + m
                    CALL DBDSQR('U', m, m, m, 0_ip_, s, work(ie), work(iu), &
                      ldwrku, work(ir), ldwrkr, dum, 1_ip_, work(iwork),    &
                      info)
                    CALL DGEMM('N', 'N', m, n, m, one, work(iu), ldwrku, a, &
                      lda, zero, vt, ldvt)
                    CALL DLACPY('F', m, m, work(ir), ldwrkr, a, lda)
                  ELSE
                    itau = 1
                    iwork = itau + m
                    CALL DGELQF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('U', m, n, a, lda, vt, ldvt)
                    CALL DORGLQ(m, n, m, vt, ldvt, work(itau), work(iwork), &
                      lwork-iwork+1, ierr)
                    ie = itau
                    itauq = ie + m
                    itaup = itauq + m
                    iwork = itaup + m
                    CALL DLASET('U', m-1, m-1, zero, zero, a(1,2), lda)
                    CALL DGEBRD(m, m, a, lda, s, work(ie), work(itauq),     &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    CALL DORMBR('P', 'L', 'T', m, n, m, a, lda,             &
                      work(itaup), vt, ldvt, work(iwork), lwork-iwork+1,    &
                      ierr)
                    CALL DORGBR('Q', m, m, m, a, lda, work(itauq),          &
                      work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + m
                    CALL DBDSQR('U', m, n, m, 0_ip_, s, work(ie), vt, ldvt, &
                      a, lda, dum, 1_ip_, work(iwork), info)
                  END IF
                ELSE IF (wntuas) THEN
                  IF (lwork>=m*m+MAX(4*m,bdspac)) THEN
                    iu = 1
                    IF (lwork>=wrkbl+lda*m) THEN
                      ldwrku = lda
                    ELSE
                      ldwrku = m
                    END IF
                    itau = iu + ldwrku*m
                    iwork = itau + m
                    CALL DGELQF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('L', m, m, a, lda, work(iu), ldwrku)
                    CALL DLASET('U', m-1, m-1, zero, zero, work(iu+ldwrku), &
                      ldwrku)
                    CALL DORGLQ(m, n, m, a, lda, work(itau), work(iwork),   &
                      lwork-iwork+1, ierr)
                    ie = itau
                    itauq = ie + m
                    itaup = itauq + m
                    iwork = itaup + m
                    CALL DGEBRD(m, m, work(iu), ldwrku, s, work(ie),        &
                      work(itauq), work(itaup), work(iwork), lwork-iwork+1, &
                      ierr)
                    CALL DLACPY('L', m, m, work(iu), ldwrku, u, ldu)
                    CALL DORGBR('P', m, m, m, work(iu), ldwrku,             &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    CALL DORGBR('Q', m, m, m, u, ldu, work(itauq),          &
                      work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + m
                    CALL DBDSQR('U', m, m, m, 0_ip_, s, work(ie), work(iu), &
                      ldwrku, u, ldu, dum, 1_ip_, work(iwork), info)
                    CALL DGEMM('N', 'N', m, n, m, one, work(iu), ldwrku, a, &
                      lda, zero, vt, ldvt)
                  ELSE
                    itau = 1
                    iwork = itau + m
                    CALL DGELQF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('U', m, n, a, lda, vt, ldvt)
                    CALL DORGLQ(m, n, m, vt, ldvt, work(itau), work(iwork), &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('L', m, m, a, lda, u, ldu)
                    CALL DLASET('U', m-1, m-1, zero, zero, u(1,2), ldu)
                    ie = itau
                    itauq = ie + m
                    itaup = itauq + m
                    iwork = itaup + m
                    CALL DGEBRD(m, m, u, ldu, s, work(ie), work(itauq),     &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    CALL DORMBR('P', 'L', 'T', m, n, m, u, ldu,             &
                      work(itaup), vt, ldvt, work(iwork), lwork-iwork+1,    &
                      ierr)
                    CALL DORGBR('Q', m, m, m, u, ldu, work(itauq),          &
                      work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + m
                    CALL DBDSQR('U', m, n, m, 0_ip_, s, work(ie), vt, ldvt, &
                      u, ldu, dum, 1_ip_, work(iwork), info)
                  END IF
                END IF
              ELSE IF (wntva) THEN
                IF (wntun) THEN
                  IF (lwork>=m*m+MAX(n+m,4*m,bdspac)) THEN
                    ir = 1
                    IF (lwork>=wrkbl+lda*m) THEN
                      ldwrkr = lda
                    ELSE
                      ldwrkr = m
                    END IF
                    itau = ir + ldwrkr*m
                    iwork = itau + m
                    CALL DGELQF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('U', m, n, a, lda, vt, ldvt)
                    CALL DLACPY('L', m, m, a, lda, work(ir), ldwrkr)
                    CALL DLASET('U', m-1, m-1, zero, zero, work(ir+ldwrkr), &
                      ldwrkr)
                    CALL DORGLQ(n, n, m, vt, ldvt, work(itau), work(iwork), &
                      lwork-iwork+1, ierr)
                    ie = itau
                    itauq = ie + m
                    itaup = itauq + m
                    iwork = itaup + m
                    CALL DGEBRD(m, m, work(ir), ldwrkr, s, work(ie),        &
                      work(itauq), work(itaup), work(iwork), lwork-iwork+1, &
                      ierr)
                    CALL DORGBR('P', m, m, m, work(ir), ldwrkr,             &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + m
                    CALL DBDSQR('U', m, m, 0_ip_, 0_ip_, s, work(ie),       &
                      work(ir), ldwrkr, dum, 1_ip_, dum, 1_ip_, work(iwork),&
                      info)
                    CALL DGEMM('N', 'N', m, n, m, one, work(ir), ldwrkr,    &
                      vt, ldvt, zero, a, lda)
                    CALL DLACPY('F', m, n, a, lda, vt, ldvt)
                  ELSE
                    itau = 1
                    iwork = itau + m
                    CALL DGELQF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('U', m, n, a, lda, vt, ldvt)
                    CALL DORGLQ(n, n, m, vt, ldvt, work(itau), work(iwork), &
                      lwork-iwork+1, ierr)
                    ie = itau
                    itauq = ie + m
                    itaup = itauq + m
                    iwork = itaup + m
                    CALL DLASET('U', m-1, m-1, zero, zero, a(1,2), lda)
                    CALL DGEBRD(m, m, a, lda, s, work(ie), work(itauq),     &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    CALL DORMBR('P', 'L', 'T', m, n, m, a, lda,             &
                      work(itaup), vt, ldvt, work(iwork), lwork-iwork+1,    &
                      ierr)
                    iwork = ie + m
                    CALL DBDSQR('U', m, n, 0_ip_, 0_ip_, s, work(ie), vt,   &
                      ldvt, dum, 1_ip_, dum, 1_ip_, work(iwork), info)
                  END IF
                ELSE IF (wntuo) THEN
                  IF (lwork>=2*m*m+MAX(n+m,4*m,bdspac)) THEN
                    iu = 1
                    IF (lwork>=wrkbl+2*lda*m) THEN
                      ldwrku = lda
                      ir = iu + ldwrku*m
                      ldwrkr = lda
                    ELSE IF (lwork>=wrkbl+(lda+m)*m) THEN
                      ldwrku = lda
                      ir = iu + ldwrku*m
                      ldwrkr = m
                    ELSE
                      ldwrku = m
                      ir = iu + ldwrku*m
                      ldwrkr = m
                    END IF
                    itau = ir + ldwrkr*m
                    iwork = itau + m
                    CALL DGELQF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('U', m, n, a, lda, vt, ldvt)
                    CALL DORGLQ(n, n, m, vt, ldvt, work(itau), work(iwork), &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('L', m, m, a, lda, work(iu), ldwrku)
                    CALL DLASET('U', m-1, m-1, zero, zero, work(iu+ldwrku), &
                      ldwrku)
                    ie = itau
                    itauq = ie + m
                    itaup = itauq + m
                    iwork = itaup + m
                    CALL DGEBRD(m, m, work(iu), ldwrku, s, work(ie),        &
                      work(itauq), work(itaup), work(iwork), lwork-iwork+1, &
                      ierr)
                    CALL DLACPY('L', m, m, work(iu), ldwrku, work(ir),      &
                      ldwrkr)
                    CALL DORGBR('P', m, m, m, work(iu), ldwrku,             &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    CALL DORGBR('Q', m, m, m, work(ir), ldwrkr,             &
                      work(itauq), work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + m
                    CALL DBDSQR('U', m, m, m, 0_ip_, s, work(ie), work(iu), &
                      ldwrku, work(ir), ldwrkr, dum, 1_ip_, work(iwork),    &
                      info)
                    CALL DGEMM('N', 'N', m, n, m, one, work(iu), ldwrku,    &
                      vt, ldvt, zero, a, lda)
                    CALL DLACPY('F', m, n, a, lda, vt, ldvt)
                    CALL DLACPY('F', m, m, work(ir), ldwrkr, a, lda)
                  ELSE
                    itau = 1
                    iwork = itau + m
                    CALL DGELQF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('U', m, n, a, lda, vt, ldvt)
                    CALL DORGLQ(n, n, m, vt, ldvt, work(itau), work(iwork), &
                      lwork-iwork+1, ierr)
                    ie = itau
                    itauq = ie + m
                    itaup = itauq + m
                    iwork = itaup + m
                    CALL DLASET('U', m-1, m-1, zero, zero, a(1,2), lda)
                    CALL DGEBRD(m, m, a, lda, s, work(ie), work(itauq),     &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    CALL DORMBR('P', 'L', 'T', m, n, m, a, lda,             &
                      work(itaup), vt, ldvt, work(iwork), lwork-iwork+1,    &
                      ierr)
                    CALL DORGBR('Q', m, m, m, a, lda, work(itauq),          &
                      work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + m
                    CALL DBDSQR('U', m, n, m, 0_ip_, s, work(ie), vt, ldvt, &
                      a, lda, dum, 1_ip_, work(iwork), info)
                  END IF
                ELSE IF (wntuas) THEN
                  IF (lwork>=m*m+MAX(n+m,4*m,bdspac)) THEN
                    iu = 1
                    IF (lwork>=wrkbl+lda*m) THEN
                      ldwrku = lda
                    ELSE
                      ldwrku = m
                    END IF
                    itau = iu + ldwrku*m
                    iwork = itau + m
                    CALL DGELQF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('U', m, n, a, lda, vt, ldvt)
                    CALL DORGLQ(n, n, m, vt, ldvt, work(itau), work(iwork), &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('L', m, m, a, lda, work(iu), ldwrku)
                    CALL DLASET('U', m-1, m-1, zero, zero, work(iu+ldwrku), &
                      ldwrku)
                    ie = itau
                    itauq = ie + m
                    itaup = itauq + m
                    iwork = itaup + m
                    CALL DGEBRD(m, m, work(iu), ldwrku, s, work(ie),        &
                      work(itauq), work(itaup), work(iwork), lwork-iwork+1, &
                      ierr)
                    CALL DLACPY('L', m, m, work(iu), ldwrku, u, ldu)
                    CALL DORGBR('P', m, m, m, work(iu), ldwrku,             &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    CALL DORGBR('Q', m, m, m, u, ldu, work(itauq),          &
                      work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + m
                    CALL DBDSQR('U', m, m, m, 0_ip_, s, work(ie), work(iu), &
                      ldwrku, u, ldu, dum, 1_ip_, work(iwork), info)
                    CALL DGEMM('N', 'N', m, n, m, one, work(iu), ldwrku,    &
                      vt, ldvt, zero, a, lda)
                    CALL DLACPY('F', m, n, a, lda, vt, ldvt)
                  ELSE
                    itau = 1
                    iwork = itau + m
                    CALL DGELQF(m, n, a, lda, work(itau), work(iwork),      &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('U', m, n, a, lda, vt, ldvt)
                    CALL DORGLQ(n, n, m, vt, ldvt, work(itau), work(iwork), &
                      lwork-iwork+1, ierr)
                    CALL DLACPY('L', m, m, a, lda, u, ldu)
                    CALL DLASET('U', m-1, m-1, zero, zero, u(1,2), ldu)
                    ie = itau
                    itauq = ie + m
                    itaup = itauq + m
                    iwork = itaup + m
                    CALL DGEBRD(m, m, u, ldu, s, work(ie), work(itauq),     &
                      work(itaup), work(iwork), lwork-iwork+1, ierr)
                    CALL DORMBR('P', 'L', 'T', m, n, m, u, ldu,             &
                      work(itaup), vt, ldvt, work(iwork), lwork-iwork+1,    &
                      ierr)
                    CALL DORGBR('Q', m, m, m, u, ldu, work(itauq),          &
                      work(iwork), lwork-iwork+1, ierr)
                    iwork = ie + m
                    CALL DBDSQR('U', m, n, m, 0_ip_, s, work(ie), vt, ldvt, &
                      u, ldu, dum, 1_ip_, work(iwork), info)
                  END IF
                END IF
              END IF
            ELSE
              ie = 1
              itauq = ie + m
              itaup = itauq + m
              iwork = itaup + m
              CALL DGEBRD(m, n, a, lda, s, work(ie), work(itauq),           &
                work(itaup), work(iwork), lwork-iwork+1, ierr)
              IF (wntuas) THEN
                CALL DLACPY('L', m, m, a, lda, u, ldu)
                CALL DORGBR('Q', m, m, n, u, ldu, work(itauq), work(iwork), &
                  lwork-iwork+1, ierr)
              END IF
              IF (wntvas) THEN
                CALL DLACPY('U', m, n, a, lda, vt, ldvt)
                IF (wntva) nrvt = n
                IF (wntvs) nrvt = m
                CALL DORGBR('P', nrvt, n, m, vt, ldvt, work(itaup),         &
                  work(iwork), lwork-iwork+1, ierr)
              END IF
              IF (wntuo) THEN
                CALL DORGBR('Q', m, m, n, a, lda, work(itauq), work(iwork), &
                  lwork-iwork+1, ierr)
              END IF
              IF (wntvo) THEN
                CALL DORGBR('P', m, n, m, a, lda, work(itaup), work(iwork), &
                  lwork-iwork+1, ierr)
              END IF
              iwork = ie + m
              IF (wntuas .OR. wntuo) nru = m
              IF (wntun) nru = 0
              IF (wntvas .OR. wntvo) ncvt = n
              IF (wntvn) ncvt = 0
              IF ((.NOT. wntuo) .AND. (.NOT. wntvo)) THEN
                CALL DBDSQR('L', m, ncvt, nru, 0_ip_, s, work(ie), vt,      &
                  ldvt, u, ldu, dum, 1_ip_, work(iwork), info)
              ELSE IF ((.NOT. wntuo) .AND. wntvo) THEN
                CALL DBDSQR('L', m, ncvt, nru, 0_ip_, s, work(ie), a, lda,  &
                  u, ldu, dum, 1_ip_, work(iwork), info)
              ELSE
                CALL DBDSQR('L', m, ncvt, nru, 0_ip_, s, work(ie), vt,      &
                  ldvt, a, lda, dum, 1_ip_, work(iwork), info)
              END IF
            END IF
          END IF
          IF (info/=0) THEN
            IF (ie>2) THEN
              DO i = 1, minmn - 1
                work(i+1) = work(i+ie-1)
              END DO
            END IF
            IF (ie<2) THEN
              DO i = minmn - 1, 1_ip_, -1_ip_
                work(i+1) = work(i+ie-1)
              END DO
            END IF
          END IF
          IF (iscl==1) THEN
            IF (anrm>bignum) CALL DLASCL('G', 0_ip_, 0_ip_, bignum, anrm,   &
              minmn, 1_ip_, s, minmn, ierr)
            IF (info/=0 .AND. anrm>bignum) CALL DLASCL('G', 0_ip_, 0_ip_,   &
              bignum, anrm, minmn-1, 1_ip_, work(2), minmn, ierr)
            IF (anrm<smlnum) CALL DLASCL('G', 0_ip_, 0_ip_, smlnum, anrm,   &
              minmn, 1_ip_, s, minmn, ierr)
            IF (info/=0 .AND. anrm<smlnum) CALL DLASCL('G', 0_ip_, 0_ip_,   &
              smlnum, anrm, minmn-1, 1_ip_, work(2), minmn, ierr)
          END IF
          work(1) = REAL(maxwrk,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DGETF2(m, n, a, lda, ipiv, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, lda, m, n
          INTEGER(ip_) :: ipiv(*)
          REAL(rp_) :: a(lda, *)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          REAL(rp_) :: sfmin
          INTEGER(ip_) :: i, j, jp
          REAL(rp_) :: DLAMCH
          INTEGER(ip_) :: IDAMAX
          EXTERNAL :: DLAMCH, IDAMAX
          EXTERNAL :: DGER, DSCAL, DSWAP, XERBLA2
          INTRINSIC :: MAX, MIN
          info = 0
          IF (m<0) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,m)) THEN
            info = -4
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GETF2', -info)
            RETURN
          END IF
          IF (m==0 .OR. n==0) RETURN
          sfmin = DLAMCH('S')
          DO j = 1, MIN(m, n)
            jp = j - 1 + IDAMAX(m-j+1, a(j,j), 1_ip_)
            ipiv(j) = jp
            IF (a(jp,j)/=zero) THEN
              IF (jp/=j) CALL DSWAP(n, a(j,1), lda, a(jp,1), lda)
              IF (j<m) THEN
                IF (ABS(a(j,j))>=sfmin) THEN
                  CALL DSCAL(m-j, one/a(j,j), a(j+1,j), 1_ip_)
                ELSE
                  DO i = 1, m - j
                    a(j+i, j) = a(j+i, j)/a(j, j)
                  END DO
                END IF
              END IF
            ELSE IF (info==0) THEN
              info = j
            END IF
            IF (j<MIN(m,n)) THEN
              CALL DGER(m-j, n-j, -one, a(j+1,j), 1_ip_, a(j,j+1), lda,     &
                a(j+1,j+1), lda)
            END IF
          END DO
          RETURN
        END SUBROUTINE

        SUBROUTINE DGETRF(m, n, a, lda, ipiv, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, lda, m, n
          INTEGER(ip_) :: ipiv(*)
          REAL(rp_) :: a(lda, *)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          INTEGER(ip_) :: i, iinfo, j, jb, nb
          EXTERNAL :: DGEMM, DGETRF2, DLASWP, DTRSM, XERBLA2
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: ILAENV2
          INTRINSIC :: MAX, MIN
          info = 0
          IF (m<0) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,m)) THEN
            info = -4
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GETRF', -info)
            RETURN
          END IF
          IF (m==0 .OR. n==0) RETURN
          nb = ILAENV2(1_ip_, 'GETRF', ' ', m, n, -1_ip_, -1_ip_)
          IF (nb<=1 .OR. nb>=MIN(m,n)) THEN
            CALL DGETRF2(m, n, a, lda, ipiv, info)
          ELSE
            DO j = 1, MIN(m, n), nb
              jb = MIN(MIN(m,n)-j+1, nb)
              CALL DGETRF2(m-j+1, jb, a(j,j), lda, ipiv(j), iinfo)
              IF (info==0 .AND. iinfo>0) info = iinfo + j - 1
              DO i = j, MIN(m, j+jb-1)
                ipiv(i) = j - 1 + ipiv(i)
              END DO
              CALL DLASWP(j-1, a, lda, j, j+jb-1, ipiv, 1_ip_)
              IF (j+jb<=n) THEN
                CALL DLASWP(n-j-jb+1, a(1,j+jb), lda, j, j+jb-1, ipiv,      &
                  1_ip_)
                CALL DTRSM('Left', 'Lower', 'No transpose', 'Unit', jb,     &
                  n-j-jb+1, one, a(j,j), lda, a(j,j+jb), lda)
                IF (j+jb<=m) THEN
                  CALL DGEMM('No transpose', 'No transpose', m-j-jb+1,      &
                    n-j-jb+1, jb, -one, a(j+jb,j), lda, a(j,j+jb), lda, one,&
                    a(j+jb,j+jb), lda)
                END IF
              END IF
            END DO
          END IF
          RETURN
        END SUBROUTINE

        RECURSIVE SUBROUTINE DGETRF2(m, n, a, lda, ipiv, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, lda, m, n
          INTEGER(ip_) :: ipiv(*)
          REAL(rp_) :: a(lda, *)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          REAL(rp_) :: sfmin, temp
          INTEGER(ip_) :: i, iinfo, n1, n2
          REAL(rp_) :: DLAMCH
          INTEGER(ip_) :: IDAMAX
          EXTERNAL :: DLAMCH, IDAMAX
          EXTERNAL :: DGEMM, DSCAL, DLASWP, DTRSM, XERBLA2
          INTRINSIC :: MAX, MIN
          info = 0
          IF (m<0) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,m)) THEN
            info = -4
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GETRF2', -info)
            RETURN
          END IF
          IF (m==0 .OR. n==0) RETURN
          IF (m==1) THEN
            ipiv(1) = 1
            IF (a(1,1)==zero) info = 1
          ELSE IF (n==1) THEN
            sfmin = DLAMCH('S')
            i = IDAMAX(m, a(1,1), 1_ip_)
            ipiv(1) = i
            IF (a(i,1)/=zero) THEN
              IF (i/=1) THEN
                temp = a(1, 1_ip_)
                a(1, 1_ip_) = a(i, 1_ip_)
                a(i, 1_ip_) = temp
              END IF
              IF (ABS(a(1,1))>=sfmin) THEN
                CALL DSCAL(m-1, one/a(1,1), a(2,1), 1_ip_)
              ELSE
                DO i = 1, m - 1
                  a(1+i, 1_ip_) = a(1+i, 1_ip_)/a(1, 1_ip_)
                END DO
              END IF
            ELSE
              info = 1
            END IF
          ELSE
            n1 = MIN(m, n)/2
            n2 = n - n1
            CALL DGETRF2(m, n1, a, lda, ipiv, iinfo)
            IF (info==0 .AND. iinfo>0) info = iinfo
            CALL DLASWP(n2, a(1,n1+1), lda, 1_ip_, n1, ipiv, 1_ip_)
            CALL DTRSM('L', 'L', 'N', 'U', n1, n2, one, a, lda, a(1,n1+1),  &
              lda)
            CALL DGEMM('N', 'N', m-n1, n2, n1, -one, a(n1+1,1), lda, a(1,   &
              n1+1), lda, one, a(n1+1,n1+1), lda)
            CALL DGETRF2(m-n1, n2, a(n1+1,n1+1), lda, ipiv(n1+1), iinfo)
            IF (info==0 .AND. iinfo>0) info = iinfo + n1
            DO i = n1 + 1, MIN(m, n)
              ipiv(i) = ipiv(i) + n1
            END DO
            CALL DLASWP(n1, a(1,1), lda, n1+1, MIN(m,n), ipiv, 1_ip_)
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DGETRS(trans, n, nrhs, a, lda, ipiv, b, ldb, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: trans
          INTEGER(ip_) :: info, lda, ldb, n, nrhs
          INTEGER(ip_) :: ipiv(*)
          REAL(rp_) :: a(lda, *), b(ldb, *)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          LOGICAL :: notran
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: DLASWP, DTRSM, XERBLA2
          INTRINSIC :: MAX
          info = 0
          notran = LSAME(trans, 'N')
          IF (.NOT. notran .AND. .NOT. LSAME(trans,'T') .AND. .NOT. &
            LSAME(trans,'C')) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (nrhs<0) THEN
            info = -3
          ELSE IF (lda<MAX(1,n)) THEN
            info = -5
          ELSE IF (ldb<MAX(1,n)) THEN
            info = -8
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GETRS', -info)
            RETURN
          END IF
          IF (n==0 .OR. nrhs==0) RETURN
          IF (notran) THEN
            CALL DLASWP(nrhs, b, ldb, 1_ip_, n, ipiv, 1_ip_)
            CALL DTRSM('Left', 'Lower', 'No transpose', 'Unit', n, nrhs,    &
              one, a, lda, b, ldb)
            CALL DTRSM('Left', 'Upper', 'No transpose', 'Non-unit', n,      &
              nrhs, one, a, lda, b, ldb)
          ELSE
            CALL DTRSM('Left', 'Upper', 'Transpose', 'Non-unit', n, nrhs,   &
              one, a, lda, b, ldb)
            CALL DTRSM('Left', 'Lower', 'Transpose', 'Unit', n, nrhs, one,  &
              a, lda, b, ldb)
            CALL DLASWP(nrhs, b, ldb, 1_ip_, n, ipiv, -1_ip_)
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DHSEQR(job, compz, n, ilo, ihi, h, ldh, wr, wi, z, ldz,  &
          work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: ihi, ilo, info, ldh, ldz, lwork, n
          CHARACTER :: compz, job
          REAL(rp_) :: h(ldh, *), wi(*), work(*), wr(*), z(ldz, *)
          INTEGER(ip_) :: ntiny
          PARAMETER (ntiny=15)
          INTEGER(ip_) :: nl
          PARAMETER (nl=49)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          REAL(rp_) :: hl(nl, nl), workl(nl)
          INTEGER(ip_) :: i, kbot, nmin
          LOGICAL :: initz, lquery, wantt, wantz
          INTEGER(ip_) :: ILAENV2
          LOGICAL :: LSAME
          EXTERNAL :: ILAENV2, LSAME
          EXTERNAL :: DLACPY, DLAHQR, DLAQR0, DLASET, &
            XERBLA2
          INTRINSIC :: REAL, MAX, MIN
          wantt = LSAME(job, 'S')
          initz = LSAME(compz, 'I')
          wantz = initz .OR. LSAME(compz, 'V')
          work(1) = REAL(MAX(1,n),rp_)
          lquery = lwork == -1
          info = 0
          IF (.NOT. LSAME(job,'E') .AND. .NOT. wantt) THEN
            info = -1
          ELSE IF (.NOT. LSAME(compz,'N') .AND. .NOT. wantz) THEN
            info = -2
          ELSE IF (n<0) THEN
            info = -3
          ELSE IF (ilo<1 .OR. ilo>MAX(1,n)) THEN
            info = -4
          ELSE IF (ihi<MIN(ilo,n) .OR. ihi>n) THEN
            info = -5
          ELSE IF (ldh<MAX(1,n)) THEN
            info = -7
          ELSE IF (ldz<1 .OR. (wantz .AND. ldz<MAX(1,n))) THEN
            info = -11
          ELSE IF (lwork<MAX(1,n) .AND. .NOT. lquery) THEN
            info = -13
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('HSEQR', -info)
            RETURN
          ELSE IF (n==0) THEN
            RETURN
          ELSE IF (lquery) THEN
            CALL DLAQR0(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, ilo,     &
              ihi, z, ldz, work, lwork, info)
            work(1) = REAL(MAX(REAL(MAX(1,n),rp_), work(1)),rp_)
            RETURN
          ELSE
            DO i = 1, ilo - 1
              wr(i) = h(i, i)
              wi(i) = zero
            END DO
            DO i = ihi + 1, n
              wr(i) = h(i, i)
              wi(i) = zero
            END DO
            IF (initz) CALL DLASET('A', n, n, zero, one, z, ldz)
            IF (ilo==ihi) THEN
              wr(ilo) = h(ilo, ilo)
              wi(ilo) = zero
              RETURN
            END IF
            nmin = ILAENV2(12_ip_, 'HSEQR', job(:1)//compz(:1), n, ilo,     &
              ihi, lwork)
            nmin = MAX(ntiny, nmin)
            IF (n>nmin) THEN
              CALL DLAQR0(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, ilo,   &
                ihi, z, ldz, work, lwork, info)
            ELSE
              CALL DLAHQR(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, ilo,   &
                ihi, z, ldz, info)
              IF (info>0) THEN
                kbot = info
                IF (n>=nl) THEN
                  CALL DLAQR0(wantt, wantz, n, ilo, kbot, h, ldh, wr, wi,   &
                    ilo, ihi, z, ldz, work, lwork, info)
                ELSE
                  CALL DLACPY('A', n, n, h, ldh, hl, nl)
                  hl(n+1, n) = zero
                  CALL DLASET('A', nl, nl-n, zero, zero, hl(1,n+1), nl)
                  CALL DLAQR0(wantt, wantz, nl, ilo, kbot, hl, nl, wr, wi,  &
                    ilo, ihi, z, ldz, workl, nl, info)
                  IF (wantt .OR. info/=0) CALL DLACPY('A', n, n, hl, nl, h, &
                    ldh)
                END IF
              END IF
            END IF
            IF ((wantt .OR. info/=0) .AND. n>2) CALL DLASET('L', n-2, n-2,  &
              zero, zero, h(3,1), ldh)
            work(1) = REAL(MAX(REAL(MAX(1,n),rp_), work(1)),rp_)
          END IF
        END SUBROUTINE

        LOGICAL FUNCTION DISNAN(din)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_), INTENT (IN) :: din
          LOGICAL :: DLAISNAN
          EXTERNAL :: DLAISNAN
          DISNAN = DLAISNAN(din, din)
          RETURN
        END FUNCTION

        SUBROUTINE DLABAD(small, large)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: large, small
          INTRINSIC :: LOG10, SQRT
          IF (LOG10(large)>2000.0_rp_) THEN
            small = SQRT(small)
            large = SQRT(large)
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLABRD(m, n, nb, a, lda, d, e, tauq, taup, x, ldx, y,    &
          ldy)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: lda, ldx, ldy, m, n, nb
          REAL(rp_) :: a(lda, *), d(*), e(*), taup(*), tauq(*), x(ldx, *),  &
            y(ldy, *)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          INTEGER(ip_) :: i
          EXTERNAL :: DGEMV, DLARFG, DSCAL
          INTRINSIC :: MIN
          IF (m<=0 .OR. n<=0) RETURN
          IF (m>=n) THEN
            DO i = 1, nb
              CALL DGEMV('No transpose', m-i+1, i-1, -one, a(i,1), lda,     &
                y(i,1), ldy, one, a(i,i), 1_ip_)
              CALL DGEMV('No transpose', m-i+1, i-1, -one, x(i,1), ldx,     &
                a(1,i), 1_ip_, one, a(i,i), 1_ip_)
              CALL DLARFG(m-i+1, a(i,i), a(MIN(i+1,m),i), 1_ip_, tauq(i))
              d(i) = a(i, i)
              IF (i<n) THEN
                a(i, i) = one
                CALL DGEMV('Transpose', m-i+1, n-i, one, a(i,i+1), lda,     &
                  a(i,i), 1_ip_, zero, y(i+1,i), 1_ip_)
                CALL DGEMV('Transpose', m-i+1, i-1, one, a(i,1), lda, a(i,  &
                  i), 1_ip_, zero, y(1,i), 1_ip_)
                CALL DGEMV('No transpose', n-i, i-1, -one, y(i+1,1), ldy,   &
                  y(1,i), 1_ip_, one, y(i+1,i), 1_ip_)
                CALL DGEMV('Transpose', m-i+1, i-1, one, x(i,1), ldx, a(i,  &
                  i), 1_ip_, zero, y(1,i), 1_ip_)
                CALL DGEMV('Transpose', i-1, n-i, -one, a(1,i+1), lda, y(1, &
                  i), 1_ip_, one, y(i+1,i), 1_ip_)
                CALL DSCAL(n-i, tauq(i), y(i+1,i), 1_ip_)
                CALL DGEMV('No transpose', n-i, i, -one, y(i+1,1), ldy,     &
                  a(i,1), lda, one, a(i,i+1), lda)
                CALL DGEMV('Transpose', i-1, n-i, -one, a(1,i+1), lda, x(i, &
                  1), ldx, one, a(i,i+1), lda)
                CALL DLARFG(n-i, a(i,i+1), a(i,MIN(i+2,n)), lda, taup(i))
                e(i) = a(i, i+1)
                a(i, i+1) = one
                CALL DGEMV('No transpose', m-i, n-i, one, a(i+1,i+1), lda,  &
                  a(i,i+1), lda, zero, x(i+1,i), 1_ip_)
                CALL DGEMV('Transpose', n-i, i, one, y(i+1,1), ldy, a(i,    &
                  i+1), lda, zero, x(1,i), 1_ip_)
                CALL DGEMV('No transpose', m-i, i, -one, a(i+1,1), lda,     &
                  x(1,i), 1_ip_, one, x(i+1,i), 1_ip_)
                CALL DGEMV('No transpose', i-1, n-i, one, a(1,i+1), lda,    &
                  a(i,i+1), lda, zero, x(1,i), 1_ip_)
                CALL DGEMV('No transpose', m-i, i-1, -one, x(i+1,1), ldx,   &
                  x(1,i), 1_ip_, one, x(i+1,i), 1_ip_)
                CALL DSCAL(m-i, taup(i), x(i+1,i), 1_ip_)
              END IF
            END DO
          ELSE
            DO i = 1, nb
              CALL DGEMV('No transpose', n-i+1, i-1, -one, y(i,1), ldy,     &
                a(i,1), lda, one, a(i,i), lda)
              CALL DGEMV('Transpose', i-1, n-i+1, -one, a(1,i), lda, x(i,   &
                1), ldx, one, a(i,i), lda)
              CALL DLARFG(n-i+1, a(i,i), a(i,MIN(i+1,n)), lda, taup(i))
              d(i) = a(i, i)
              IF (i<m) THEN
                a(i, i) = one
                CALL DGEMV('No transpose', m-i, n-i+1, one, a(i+1,i), lda,  &
                  a(i,i), lda, zero, x(i+1,i), 1_ip_)
                CALL DGEMV('Transpose', n-i+1, i-1, one, y(i,1), ldy, a(i,  &
                  i), lda, zero, x(1,i), 1_ip_)
                CALL DGEMV('No transpose', m-i, i-1, -one, a(i+1,1), lda,   &
                  x(1,i), 1_ip_, one, x(i+1,i), 1_ip_)
                CALL DGEMV('No transpose', i-1, n-i+1, one, a(1,i), lda,    &
                  a(i,i), lda, zero, x(1,i), 1_ip_)
                CALL DGEMV('No transpose', m-i, i-1, -one, x(i+1,1), ldx,   &
                  x(1,i), 1_ip_, one, x(i+1,i), 1_ip_)
                CALL DSCAL(m-i, taup(i), x(i+1,i), 1_ip_)
                CALL DGEMV('No transpose', m-i, i-1, -one, a(i+1,1), lda,   &
                  y(i,1), ldy, one, a(i+1,i), 1_ip_)
                CALL DGEMV('No transpose', m-i, i, -one, x(i+1,1), ldx,     &
                  a(1,i), 1_ip_, one, a(i+1,i), 1_ip_)
                CALL DLARFG(m-i, a(i+1,i), a(MIN(i+2,m),i), 1_ip_, tauq(i))
                e(i) = a(i+1, i)
                a(i+1, i) = one
                CALL DGEMV('Transpose', m-i, n-i, one, a(i+1,i+1), lda,     &
                  a(i+1,i), 1_ip_, zero, y(i+1,i), 1_ip_)
                CALL DGEMV('Transpose', m-i, i-1, one, a(i+1,1), lda,       &
                  a(i+1,i), 1_ip_, zero, y(1,i), 1_ip_)
                CALL DGEMV('No transpose', n-i, i-1, -one, y(i+1,1), ldy,   &
                  y(1,i), 1_ip_, one, y(i+1,i), 1_ip_)
                CALL DGEMV('Transpose', m-i, i, one, x(i+1,1), ldx, a(i+1,  &
                  i), 1_ip_, zero, y(1,i), 1_ip_)
                CALL DGEMV('Transpose', i, n-i, -one, a(1,i+1), lda, y(1,   &
                  i), 1_ip_, one, y(i+1,i), 1_ip_)
                CALL DSCAL(n-i, tauq(i), y(i+1,i), 1_ip_)
              END IF
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLACPY(uplo, m, n, a, lda, b, ldb)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: lda, ldb, m, n
          REAL(rp_) :: a(lda, *), b(ldb, *)
          INTEGER(ip_) :: i, j
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          INTRINSIC :: MIN
          IF (LSAME(uplo,'U')) THEN
            DO j = 1, n
              DO i = 1, MIN(j, m)
                b(i, j) = a(i, j)
              END DO
            END DO
          ELSE IF (LSAME(uplo,'L')) THEN
            DO j = 1, n
              DO i = j, m
                b(i, j) = a(i, j)
              END DO
            END DO
          ELSE
            DO j = 1, n
              DO i = 1, m
                b(i, j) = a(i, j)
              END DO
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLAE2(a, b, c, rt1, rt2)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: a, b, c, rt1, rt2
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          REAL(rp_) :: two
          PARAMETER (two=2.0_rp_)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          REAL(rp_) :: half
          PARAMETER (half=0.5_rp_)
          REAL(rp_) :: ab, acmn, acmx, adf, df, rt, sm, tb
          INTRINSIC :: ABS, SQRT
          sm = a + c
          df = a - c
          adf = ABS(df)
          tb = b + b
          ab = ABS(tb)
          IF (ABS(a)>ABS(c)) THEN
            acmx = a
            acmn = c
          ELSE
            acmx = c
            acmn = a
          END IF
          IF (adf>ab) THEN
            rt = adf*SQRT(one+(ab/adf)**2)
          ELSE IF (adf<ab) THEN
            rt = ab*SQRT(one+(adf/ab)**2)
          ELSE
            rt = ab*SQRT(two)
          END IF
          IF (sm<zero) THEN
            rt1 = half*(sm-rt)
            rt2 = (acmx/rt1)*acmn - (b/rt1)*b
          ELSE IF (sm>zero) THEN
            rt1 = half*(sm+rt)
            rt2 = (acmx/rt1)*acmn - (b/rt1)*b
          ELSE
            rt1 = half*rt
            rt2 = -half*rt
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLAED6(kniter, orgati, rho, d, z, finit, tau, info)
          USE BLAS_LAPACK_KINDS_precision
          LOGICAL :: orgati
          INTEGER(ip_) :: info, kniter
          REAL(rp_) :: finit, rho, tau
          REAL(rp_) :: d(3), z(3)
          INTEGER(ip_) :: maxit
          PARAMETER (maxit=40)
          REAL(rp_) :: zero, one, two, three, four, eight
          PARAMETER (zero=0.0_rp_, one=1.0_rp_, two=2.0_rp_, three=3.0_rp_, &
            four=4.0_rp_, eight=8.0_rp_)
          REAL(rp_) :: DLAMCH
          EXTERNAL :: DLAMCH
          REAL(rp_) :: dscale(3), zscale(3)
          LOGICAL :: scale
          INTEGER(ip_) :: i, iter, niter
          REAL(rp_) :: a, b, base, c, ddf, df, eps, erretm, eta, f, fc,     &
            sclfac, sclinv, small1, small2, sminv1, sminv2, temp, temp1,    &
            temp2, temp3, temp4, lbd, ubd
          INTRINSIC :: ABS, INT, LOG, MAX, MIN, SQRT
          info = 0
          IF (orgati) THEN
            lbd = d(2)
            ubd = d(3)
          ELSE
            lbd = d(1)
            ubd = d(2)
          END IF
          IF (finit<zero) THEN
            lbd = zero
          ELSE
            ubd = zero
          END IF
          niter = 1
          tau = zero
          IF (kniter==2) THEN
            IF (orgati) THEN
              temp = (d(3)-d(2))/two
              c = rho + z(1)/((d(1)-d(2))-temp)
              a = c*(d(2)+d(3)) + z(2) + z(3)
              b = c*d(2)*d(3) + z(2)*d(3) + z(3)*d(2)
            ELSE
              temp = (d(1)-d(2))/two
              c = rho + z(3)/((d(3)-d(2))-temp)
              a = c*(d(1)+d(2)) + z(1) + z(2)
              b = c*d(1)*d(2) + z(1)*d(2) + z(2)*d(1)
            END IF
            temp = MAX(ABS(a), ABS(b), ABS(c))
            a = a/temp
            b = b/temp
            c = c/temp
            IF (c==zero) THEN
              tau = b/a
            ELSE IF (a<=zero) THEN
              tau = (a-SQRT(ABS(a*a-four*b*c)))/(two*c)
            ELSE
              tau = two*b/(a+SQRT(ABS(a*a-four*b*c)))
            END IF
            IF (tau<lbd .OR. tau>ubd) tau = (lbd+ubd)/two
            IF (d(1)==tau .OR. d(2)==tau .OR. d(3)==tau) THEN
              tau = zero
            ELSE
              temp = finit + tau*z(1)/(d(1)*(d(1)-tau)) +                   &
                tau*z(2)/(d(2)*(d(2)-tau)) + tau*z(3)/(d(3)*(d(3)-tau))
              IF (temp<=zero) THEN
                lbd = tau
              ELSE
                ubd = tau
              END IF
              IF (ABS(finit)<=ABS(temp)) tau = zero
            END IF
          END IF
          eps = DLAMCH('Epsilon')
          base = DLAMCH('Base')
          small1 = base**(INT(LOG(DLAMCH('SafMin'))/LOG(base)/three))
          sminv1 = one/small1
          small2 = small1*small1
          sminv2 = sminv1*sminv1
          IF (orgati) THEN
            temp = MIN(ABS(d(2)-tau), ABS(d(3)-tau))
          ELSE
            temp = MIN(ABS(d(1)-tau), ABS(d(2)-tau))
          END IF
          scale = .FALSE.
          IF (temp<=small1) THEN
            scale = .TRUE.
            IF (temp<=small2) THEN
              sclfac = sminv2
              sclinv = small2
            ELSE
              sclfac = sminv1
              sclinv = small1
            END IF
            DO i = 1, 3
              dscale(i) = d(i)*sclfac
              zscale(i) = z(i)*sclfac
            END DO
            tau = tau*sclfac
            lbd = lbd*sclfac
            ubd = ubd*sclfac
          ELSE
            DO i = 1, 3
              dscale(i) = d(i)
              zscale(i) = z(i)
            END DO
          END IF
          fc = zero
          df = zero
          ddf = zero
          DO i = 1, 3
            temp = one/(dscale(i)-tau)
            temp1 = zscale(i)*temp
            temp2 = temp1*temp
            temp3 = temp2*temp
            fc = fc + temp1/dscale(i)
            df = df + temp2
            ddf = ddf + temp3
          END DO
          f = finit + tau*fc
          IF (ABS(f)<=zero) GO TO 60
          IF (f<=zero) THEN
            lbd = tau
          ELSE
            ubd = tau
          END IF
          iter = niter + 1
          DO niter = iter, maxit
            IF (orgati) THEN
              temp1 = dscale(2) - tau
              temp2 = dscale(3) - tau
            ELSE
              temp1 = dscale(1) - tau
              temp2 = dscale(2) - tau
            END IF
            a = (temp1+temp2)*f - temp1*temp2*df
            b = temp1*temp2*f
            c = f - (temp1+temp2)*df + temp1*temp2*ddf
            temp = MAX(ABS(a), ABS(b), ABS(c))
            a = a/temp
            b = b/temp
            c = c/temp
            IF (c==zero) THEN
              eta = b/a
            ELSE IF (a<=zero) THEN
              eta = (a-SQRT(ABS(a*a-four*b*c)))/(two*c)
            ELSE
              eta = two*b/(a+SQRT(ABS(a*a-four*b*c)))
            END IF
            IF (f*eta>=zero) THEN
              eta = -f/df
            END IF
            tau = tau + eta
            IF (tau<lbd .OR. tau>ubd) tau = (lbd+ubd)/two
            fc = zero
            erretm = zero
            df = zero
            ddf = zero
            DO i = 1, 3
              IF ((dscale(i)-tau)/=zero) THEN
                temp = one/(dscale(i)-tau)
                temp1 = zscale(i)*temp
                temp2 = temp1*temp
                temp3 = temp2*temp
                temp4 = temp1/dscale(i)
                fc = fc + temp4
                erretm = erretm + ABS(temp4)
                df = df + temp2
                ddf = ddf + temp3
              ELSE
                GO TO 60
              END IF
            END DO
            f = finit + tau*fc
            erretm = eight*(ABS(finit)+ABS(tau)*erretm) + ABS(tau)*df
            IF ((ABS(f)<=four*eps*erretm) .OR. ((ubd-                       &
              lbd)<=four*eps*ABS(tau))) GO TO 60
            IF (f<=zero) THEN
              lbd = tau
            ELSE
              ubd = tau
            END IF
          END DO
          info = 1
 60       CONTINUE
          IF (scale) tau = tau*sclinv
          RETURN
        END SUBROUTINE

        SUBROUTINE DLAEV2(a, b, c, rt1, rt2, cs1, sn1)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: a, b, c, cs1, rt1, rt2, sn1
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          REAL(rp_) :: two
          PARAMETER (two=2.0_rp_)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          REAL(rp_) :: half
          PARAMETER (half=0.5_rp_)
          INTEGER(ip_) :: sgn1, sgn2
          REAL(rp_) :: ab, acmn, acmx, acs, adf, cs, ct, df, rt, sm, tb, tn
          INTRINSIC :: ABS, SQRT
          sm = a + c
          df = a - c
          adf = ABS(df)
          tb = b + b
          ab = ABS(tb)
          IF (ABS(a)>ABS(c)) THEN
            acmx = a
            acmn = c
          ELSE
            acmx = c
            acmn = a
          END IF
          IF (adf>ab) THEN
            rt = adf*SQRT(one+(ab/adf)**2)
          ELSE IF (adf<ab) THEN
            rt = ab*SQRT(one+(adf/ab)**2)
          ELSE
            rt = ab*SQRT(two)
          END IF
          IF (sm<zero) THEN
            rt1 = half*(sm-rt)
            sgn1 = -1
            rt2 = (acmx/rt1)*acmn - (b/rt1)*b
          ELSE IF (sm>zero) THEN
            rt1 = half*(sm+rt)
            sgn1 = 1
            rt2 = (acmx/rt1)*acmn - (b/rt1)*b
          ELSE
            rt1 = half*rt
            rt2 = -half*rt
            sgn1 = 1
          END IF
          IF (df>=zero) THEN
            cs = df + rt
            sgn2 = 1
          ELSE
            cs = df - rt
            sgn2 = -1
          END IF
          acs = ABS(cs)
          IF (acs>ab) THEN
            ct = -tb/cs
            sn1 = one/SQRT(one+ct*ct)
            cs1 = ct*sn1
          ELSE
            IF (ab==zero) THEN
              cs1 = one
              sn1 = zero
            ELSE
              tn = -cs/tb
              cs1 = one/SQRT(one+tn*tn)
              sn1 = tn*cs1
            END IF
          END IF
          IF (sgn1==sgn2) THEN
            tn = cs1
            cs1 = -sn1
            sn1 = tn
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLAEXC(wantq, n, t, ldt, q, ldq, j1, n1, n2, work, info)
          USE BLAS_LAPACK_KINDS_precision
          LOGICAL :: wantq
          INTEGER(ip_) :: info, j1, ldq, ldt, n, n1, n2
          REAL(rp_) :: q(ldq, *), t(ldt, *), work(*)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          REAL(rp_) :: ten
          PARAMETER (ten=1.0D+1)
          INTEGER(ip_) :: ldd, ldx
          PARAMETER (ldd=4, ldx=2)
          INTEGER(ip_) :: ierr, j2, j3, j4, k, nd
          REAL(rp_) :: cs, dnorm, eps, scale, smlnum, sn, t11, t22, t33,    &
            tau, tau1, tau2, temp, thresh, wi1, wi2, wr1, wr2, xnorm
          REAL(rp_) :: d(ldd, 4_ip_), u(3), u1(3), u2(3), x(ldx, 2_ip_)
          REAL(rp_) :: DLAMCH, DLANGE
          EXTERNAL :: DLAMCH, DLANGE
          EXTERNAL :: DLACPY, DLANV2, DLARFG, DLARFX, &
            DLARTG, DLASY2, DROT
          INTRINSIC :: ABS, MAX
          info = 0
          IF (n==0 .OR. n1==0 .OR. n2==0) RETURN
          IF (j1+n1>n) RETURN
          j2 = j1 + 1
          j3 = j1 + 2
          j4 = j1 + 3
          IF (n1==1 .AND. n2==1) THEN
            t11 = t(j1, j1)
            t22 = t(j2, j2)
            CALL DLARTG(t(j1,j2), t22-t11, cs, sn, temp)
            IF (j3<=n) CALL DROT(n-j1-1, t(j1,j3), ldt, t(j2,j3), ldt, cs,  &
              sn)
            CALL DROT(j1-1, t(1,j1), 1_ip_, t(1,j2), 1_ip_, cs, sn)
            t(j1, j1) = t22
            t(j2, j2) = t11
            IF (wantq) THEN
              CALL DROT(n, q(1,j1), 1_ip_, q(1,j2), 1_ip_, cs, sn)
            END IF
          ELSE
            nd = n1 + n2
            CALL DLACPY('Full', nd, nd, t(j1,j1), ldt, d, ldd)
            dnorm = DLANGE('Max', nd, nd, d, ldd, work)
            eps = DLAMCH('P')
            smlnum = DLAMCH('S')/eps
            thresh = MAX(ten*eps*dnorm, smlnum)
            CALL DLASY2(.FALSE., .FALSE., -1_ip_, n1, n2, d, ldd, d(n1+1,   &
              n1+1), ldd, d(1,n1+1), ldd, scale, x, ldx, xnorm, ierr)
            k = n1 + n1 + n2 - 3
            GO TO (10, 20, 30) k
 10         CONTINUE
            u(1) = scale
            u(2) = x(1, 1_ip_)
            u(3) = x(1, 2_ip_)
            CALL DLARFG( 3_ip_, u(3), u, 1_ip_, tau)
            u(3) = one
            t11 = t(j1, j1)
            CALL DLARFX('L', 3_ip_, 3_ip_, u, tau, d, ldd, work)
            CALL DLARFX('R', 3_ip_, 3_ip_, u, tau, d, ldd, work)
            IF (MAX(ABS(d(3,1)),ABS(d(3,2)),ABS(d(3,3)- t11))>thresh) GO    &
              TO 50
            CALL DLARFX('L', 3_ip_, n-j1+1, u, tau, t(j1,j1), ldt, work)
            CALL DLARFX('R', j2, 3_ip_, u, tau, t(1,j1), ldt, work)
            t(j3, j1) = zero
            t(j3, j2) = zero
            t(j3, j3) = t11
            IF (wantq) THEN
              CALL DLARFX('R', n, 3_ip_, u, tau, q(1,j1), ldq, work)
            END IF
            GO TO 40
 20         CONTINUE
            u(1) = -x(1, 1_ip_)
            u(2) = -x( 2_ip_, 1_ip_)
            u(3) = scale
            CALL DLARFG( 3_ip_, u(1), u(2), 1_ip_, tau)
            u(1) = one
            t33 = t(j3, j3)
            CALL DLARFX('L', 3_ip_, 3_ip_, u, tau, d, ldd, work)
            CALL DLARFX('R', 3_ip_, 3_ip_, u, tau, d, ldd, work)
            IF (MAX(ABS(d(2,1)),ABS(d(3,1)),ABS(d(1,1)- t33))>thresh) GO    &
              TO 50
            CALL DLARFX('R', j3, 3_ip_, u, tau, t(1,j1), ldt, work)
            CALL DLARFX('L', 3_ip_, n-j1, u, tau, t(j1,j2), ldt, work)
            t(j1, j1) = t33
            t(j2, j1) = zero
            t(j3, j1) = zero
            IF (wantq) THEN
              CALL DLARFX('R', n, 3_ip_, u, tau, q(1,j1), ldq, work)
            END IF
            GO TO 40
 30         CONTINUE
            u1(1) = -x(1, 1_ip_)
            u1(2) = -x( 2_ip_, 1_ip_)
            u1(3) = scale
            CALL DLARFG( 3_ip_, u1(1), u1(2), 1_ip_, tau1)
            u1(1) = one
            temp = -tau1*(x(1,2)+u1(2)*x(2,2))
            u2(1) = -temp*u1(2) - x( 2_ip_, 2_ip_)
            u2(2) = -temp*u1(3)
            u2(3) = scale
            CALL DLARFG( 3_ip_, u2(1), u2(2), 1_ip_, tau2)
            u2(1) = one
            CALL DLARFX('L', 3_ip_, 4_ip_, u1, tau1, d, ldd, work)
            CALL DLARFX('R', 4_ip_, 3_ip_, u1, tau1, d, ldd, work)
            CALL DLARFX('L', 3_ip_, 4_ip_, u2, tau2, d(2,1), ldd, work)
            CALL DLARFX('R', 4_ip_, 3_ip_, u2, tau2, d(1,2), ldd, work)
            IF (MAX(ABS(d(3,1)),ABS(d(3,2)),ABS(d(4,1)),ABS( d(4,           &
              2)))>thresh) GO TO 50
            CALL DLARFX('L', 3_ip_, n-j1+1, u1, tau1, t(j1,j1), ldt, work)
            CALL DLARFX('R', j4, 3_ip_, u1, tau1, t(1,j1), ldt, work)
            CALL DLARFX('L', 3_ip_, n-j1+1, u2, tau2, t(j2,j1), ldt, work)
            CALL DLARFX('R', j4, 3_ip_, u2, tau2, t(1,j2), ldt, work)
            t(j3, j1) = zero
            t(j3, j2) = zero
            t(j4, j1) = zero
            t(j4, j2) = zero
            IF (wantq) THEN
              CALL DLARFX('R', n, 3_ip_, u1, tau1, q(1,j1), ldq, work)
              CALL DLARFX('R', n, 3_ip_, u2, tau2, q(1,j2), ldq, work)
            END IF
 40         CONTINUE
            IF (n2==2) THEN
              CALL DLANV2(t(j1,j1), t(j1,j2), t(j2,j1), t(j2,j2), wr1, wi1, &
                wr2, wi2, cs, sn)
              CALL DROT(n-j1-1, t(j1,j1+2), ldt, t(j2,j1+2), ldt, cs, sn)
              CALL DROT(j1-1, t(1,j1), 1_ip_, t(1,j2), 1_ip_, cs, sn)
              IF (wantq) CALL DROT(n, q(1,j1), 1_ip_, q(1,j2), 1_ip_, cs,   &
                sn)
            END IF
            IF (n1==2) THEN
              j3 = j1 + n2
              j4 = j3 + 1
              CALL DLANV2(t(j3,j3), t(j3,j4), t(j4,j3), t(j4,j4), wr1, wi1, &
                wr2, wi2, cs, sn)
              IF (j3+2<=n) CALL DROT(n-j3-1, t(j3,j3+2), ldt, t(j4,j3+2),   &
                ldt, cs, sn)
              CALL DROT(j3-1, t(1,j3), 1_ip_, t(1,j4), 1_ip_, cs, sn)
              IF (wantq) CALL DROT(n, q(1,j3), 1_ip_, q(1,j4), 1_ip_, cs,   &
                sn)
            END IF
          END IF
          RETURN
 50       CONTINUE
          info = 1
          RETURN
        END SUBROUTINE

        SUBROUTINE DLAHQR(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, iloz,  &
          ihiz, z, ldz, info)
          USE BLAS_LAPACK_KINDS_precision
          IMPLICIT NONE
          INTEGER(ip_) :: ihi, ihiz, ilo, iloz, info, ldh, ldz, n
          LOGICAL :: wantt, wantz
          REAL(rp_) :: h(ldh, *), wi(*), wr(*), z(ldz, *)
          REAL(rp_) :: zero, one, two
          PARAMETER (zero=0.0_rp_, one=1.0_rp_, two=2.0_rp_)
          REAL(rp_) :: dat1, dat2
          PARAMETER (dat1=3.0_rp_/4.0_rp_, dat2=-0.4375_rp_)
          INTEGER(ip_) :: kexsh
          PARAMETER (kexsh=10)
          REAL(rp_) :: aa, ab, ba, bb, cs, det, h11, h12, h21, h21s, h22,   &
            rt1i, rt1r, rt2i, rt2r, rtdisc, s, safmax, safmin, smlnum, sn,  &
            sum, t1, t2, t3, tr, tst, ulp, v2, v3
          INTEGER(ip_) :: i, i1, i2, its, itmax, j, k, l, m, nh, nr, nz,    &
            kdefl
          REAL(rp_) :: v(3)
          REAL(rp_) :: DLAMCH
          EXTERNAL :: DLAMCH
          EXTERNAL :: DCOPY, DLABAD, DLANV2, DLARFG, DROT
          INTRINSIC :: ABS, REAL, MAX, MIN, SQRT
          info = 0
          IF (n==0) RETURN
          IF (ilo==ihi) THEN
            wr(ilo) = h(ilo, ilo)
            wi(ilo) = zero
            RETURN
          END IF
          DO j = ilo, ihi - 3
            h(j+2, j) = zero
            h(j+3, j) = zero
          END DO
          IF (ilo<=ihi-2) h(ihi, ihi-2) = zero
          nh = ihi - ilo + 1
          nz = ihiz - iloz + 1
          safmin = DLAMCH('SAFE MINIMUM')
          safmax = one/safmin
          CALL DLABAD(safmin, safmax)
          ulp = DLAMCH('PRECISION')
          smlnum = safmin*(REAL(nh,rp_)/ulp)
          IF (wantt) THEN
            i1 = 1
            i2 = n
          END IF
          itmax = 30*MAX(10, nh)
          kdefl = 0
          i = ihi
 20       CONTINUE
          l = ilo
          IF (i<ilo) GO TO 160
          DO its = 0, itmax
            DO k = i, l + 1, -1_ip_
              IF (ABS(h(k,k-1))<=smlnum) GO TO 40
              tst = ABS(h(k-1,k-1)) + ABS(h(k,k))
              IF (tst==zero) THEN
                IF (k-2>=ilo) tst = tst + ABS(h(k-1,k-2))
                IF (k+1<=ihi) tst = tst + ABS(h(k+1,k))
              END IF
              IF (ABS(h(k,k-1))<=ulp*tst) THEN
                ab = MAX(ABS(h(k,k-1)), ABS(h(k-1,k)))
                ba = MIN(ABS(h(k,k-1)), ABS(h(k-1,k)))
                aa = MAX(ABS(h(k,k)), ABS(h(k-1,k-1)-h(k,k)))
                bb = MIN(ABS(h(k,k)), ABS(h(k-1,k-1)-h(k,k)))
                s = aa + ab
                IF (ba*(ab/s)<=MAX(smlnum,ulp*(bb*(aa/s)))) GO TO 40
              END IF
            END DO
 40         CONTINUE
            l = k
            IF (l>ilo) THEN
              h(l, l-1) = zero
            END IF
            IF (l>=i-1) GO TO 150
            kdefl = kdefl + 1
            IF (.NOT. wantt) THEN
              i1 = l
              i2 = i
            END IF
            IF (MOD(kdefl,2*kexsh)==0) THEN
              s = ABS(h(i,i-1)) + ABS(h(i-1,i-2))
              h11 = dat1*s + h(i, i)
              h12 = dat2*s
              h21 = s
              h22 = h11
            ELSE IF (MOD(kdefl,kexsh)==0) THEN
              s = ABS(h(l+1,l)) + ABS(h(l+2,l+1))
              h11 = dat1*s + h(l, l)
              h12 = dat2*s
              h21 = s
              h22 = h11
            ELSE
              h11 = h(i-1, i-1)
              h21 = h(i, i-1)
              h12 = h(i-1, i)
              h22 = h(i, i)
            END IF
            s = ABS(h11) + ABS(h12) + ABS(h21) + ABS(h22)
            IF (s==zero) THEN
              rt1r = zero
              rt1i = zero
              rt2r = zero
              rt2i = zero
            ELSE
              h11 = h11/s
              h21 = h21/s
              h12 = h12/s
              h22 = h22/s
              tr = (h11+h22)/two
              det = (h11-tr)*(h22-tr) - h12*h21
              rtdisc = SQRT(ABS(det))
              IF (det>=zero) THEN
                rt1r = tr*s
                rt2r = rt1r
                rt1i = rtdisc*s
                rt2i = -rt1i
              ELSE
                rt1r = tr + rtdisc
                rt2r = tr - rtdisc
                IF (ABS(rt1r-h22)<=ABS(rt2r-h22)) THEN
                  rt1r = rt1r*s
                  rt2r = rt1r
                ELSE
                  rt2r = rt2r*s
                  rt1r = rt2r
                END IF
                rt1i = zero
                rt2i = zero
              END IF
            END IF
            DO m = i - 2, l, -1_ip_
              h21s = h(m+1, m)
              s = ABS(h(m,m)-rt2r) + ABS(rt2i) + ABS(h21s)
              h21s = h(m+1, m)/s
              v(1) = h21s*h(m, m+1) + (h(m,m)-rt1r)*((h(m, m)-rt2r)/s) -    &
                rt1i*(rt2i/s)
              v(2) = h21s*(h(m,m)+h(m+1,m+1)-rt1r-rt2r)
              v(3) = h21s*h(m+2, m+1)
              s = ABS(v(1)) + ABS(v(2)) + ABS(v(3))
              v(1) = v(1)/s
              v(2) = v(2)/s
              v(3) = v(3)/s
              IF (m==l) GO TO 60
              IF (ABS(h(m,m-1))*(ABS(v(2))+ABS(v(3)))<=ulp*ABS(v(1))*(ABS(  &
                h(m-1,m-1))+ABS(h(m,m))+ABS(h(m+1,m+1)))) GO TO 60
            END DO
 60         CONTINUE
            DO k = m, i - 1
              nr = MIN( 3_ip_, i-k+1)
              IF (k>m) CALL DCOPY(nr, h(k,k-1), 1_ip_, v, 1_ip_)
              CALL DLARFG(nr, v(1), v(2), 1_ip_, t1)
              IF (k>m) THEN
                h(k, k-1) = v(1)
                h(k+1, k-1) = zero
                IF (k<i-1) h(k+2, k-1) = zero
              ELSE IF (m>l) THEN
                h(k, k-1) = h(k, k-1)*(one-t1)
              END IF
              v2 = v(2)
              t2 = t1*v2
              IF (nr==3) THEN
                v3 = v(3)
                t3 = t1*v3
                DO j = k, i2
                  sum = h(k, j) + v2*h(k+1, j) + v3*h(k+2, j)
                  h(k, j) = h(k, j) - sum*t1
                  h(k+1, j) = h(k+1, j) - sum*t2
                  h(k+2, j) = h(k+2, j) - sum*t3
                END DO
                DO j = i1, MIN(k+3, i)
                  sum = h(j, k) + v2*h(j, k+1) + v3*h(j, k+2)
                  h(j, k) = h(j, k) - sum*t1
                  h(j, k+1) = h(j, k+1) - sum*t2
                  h(j, k+2) = h(j, k+2) - sum*t3
                END DO
                IF (wantz) THEN
                  DO j = iloz, ihiz
                    sum = z(j, k) + v2*z(j, k+1) + v3*z(j, k+2)
                    z(j, k) = z(j, k) - sum*t1
                    z(j, k+1) = z(j, k+1) - sum*t2
                    z(j, k+2) = z(j, k+2) - sum*t3
                  END DO
                END IF
              ELSE IF (nr==2) THEN
                DO j = k, i2
                  sum = h(k, j) + v2*h(k+1, j)
                  h(k, j) = h(k, j) - sum*t1
                  h(k+1, j) = h(k+1, j) - sum*t2
                END DO
                DO j = i1, i
                  sum = h(j, k) + v2*h(j, k+1)
                  h(j, k) = h(j, k) - sum*t1
                  h(j, k+1) = h(j, k+1) - sum*t2
                END DO
                IF (wantz) THEN
                  DO j = iloz, ihiz
                    sum = z(j, k) + v2*z(j, k+1)
                    z(j, k) = z(j, k) - sum*t1
                    z(j, k+1) = z(j, k+1) - sum*t2
                  END DO
                END IF
              END IF
            END DO
          END DO
          info = i
          RETURN
 150      CONTINUE
          IF (l==i) THEN
            wr(i) = h(i, i)
            wi(i) = zero
          ELSE IF (l==i-1) THEN
            CALL DLANV2(h(i-1,i-1), h(i-1,i), h(i,i-1), h(i,i), wr(i-1),    &
              wi(i-1), wr(i), wi(i), cs, sn)
            IF (wantt) THEN
              IF (i2>i) CALL DROT(i2-i, h(i-1,i+1), ldh, h(i,i+1), ldh, cs, &
                sn)
              CALL DROT(i-i1-1, h(i1,i-1), 1_ip_, h(i1,i), 1_ip_, cs, sn)
            END IF
            IF (wantz) THEN
              CALL DROT(nz, z(iloz,i-1), 1_ip_, z(iloz,i), 1_ip_, cs, sn)
            END IF
          END IF
          kdefl = 0
          i = l - 1
          GO TO 20
 160      CONTINUE
          RETURN
        END SUBROUTINE

        SUBROUTINE DLAHR2(n, k, nb, a, lda, tau, t, ldt, y, ldy)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: k, lda, ldt, ldy, n, nb
          REAL(rp_) :: a(lda, *), t(ldt, nb), tau(nb), y(ldy, nb)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          INTEGER(ip_) :: i
          REAL(rp_) :: ei
          EXTERNAL :: DAXPY, DCOPY, DGEMM, DGEMV, DLACPY, &
            DLARFG, DSCAL, DTRMM, DTRMV
          INTRINSIC :: MIN
          IF (n<=1) RETURN
          DO i = 1, nb
            IF (i>1) THEN
              CALL DGEMV('NO TRANSPOSE', n-k, i-1, -one, y(k+1,1), ldy,     &
                a(k+i-1,1), lda, one, a(k+1,i), 1_ip_)
              CALL DCOPY(i-1, a(k+1,i), 1_ip_, t(1,nb), 1_ip_)
              CALL DTRMV('Lower', 'Transpose', 'UNIT', i-1, a(k+1,1), lda,  &
                t(1,nb), 1_ip_)
              CALL DGEMV('Transpose', n-k-i+1, i-1, one, a(k+i,1), lda,     &
                a(k+i,i), 1_ip_, one, t(1,nb), 1_ip_)
              CALL DTRMV('Upper', 'Transpose', 'NON-UNIT', i-1, t, ldt,     &
                t(1,nb), 1_ip_)
              CALL DGEMV('NO TRANSPOSE', n-k-i+1, i-1, -one, a(k+i,1), lda, &
                t(1,nb), 1_ip_, one, a(k+i,i), 1_ip_)
              CALL DTRMV('Lower', 'NO TRANSPOSE', 'UNIT', i-1, a(k+1,1),    &
                lda, t(1,nb), 1_ip_)
              CALL DAXPY(i-1, -one, t(1,nb), 1_ip_, a(k+1,i), 1_ip_)
              a(k+i-1, i-1) = ei
            END IF
            CALL DLARFG(n-k-i+1, a(k+i,i), a(MIN(k+i+1,n),i), 1_ip_,        &
              tau(i))
            ei = a(k+i, i)
            a(k+i, i) = one
            CALL DGEMV('NO TRANSPOSE', n-k, n-k-i+1, one, a(k+1,i+1), lda,  &
              a(k+i,i), 1_ip_, zero, y(k+1,i), 1_ip_)
            CALL DGEMV('Transpose', n-k-i+1, i-1, one, a(k+i,1), lda,       &
              a(k+i,i), 1_ip_, zero, t(1,i), 1_ip_)
            CALL DGEMV('NO TRANSPOSE', n-k, i-1, -one, y(k+1,1), ldy, t(1,  &
              i), 1_ip_, one, y(k+1,i), 1_ip_)
            CALL DSCAL(n-k, tau(i), y(k+1,i), 1_ip_)
            CALL DSCAL(i-1, -tau(i), t(1,i), 1_ip_)
            CALL DTRMV('Upper', 'No Transpose', 'NON-UNIT', i-1, t, ldt,    &
              t(1,i), 1_ip_)
            t(i, i) = tau(i)
          END DO
          a(k+nb, nb) = ei
          CALL DLACPY('ALL', k, nb, a(1,2), lda, y, ldy)
          CALL DTRMM('RIGHT', 'Lower', 'NO TRANSPOSE', 'UNIT', k, nb, one,  &
            a(k+1,1), lda, y, ldy)
          IF (n>k+nb) CALL DGEMM('NO TRANSPOSE', 'NO TRANSPOSE', k, nb,     &
            n-k-nb, one, a(1,2+nb), lda, a(k+1+nb,1), lda, one, y, ldy)
          CALL DTRMM('RIGHT', 'Upper', 'NO TRANSPOSE', 'NON-UNIT', k, nb,   &
            one, t, ldt, y, ldy)
          RETURN
        END SUBROUTINE

        SUBROUTINE DLAIC1(job, j, x, sest, w, gamma, sestpr, s, c)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: j, job
          REAL(rp_) :: c, gamma, s, sest, sestpr
          REAL(rp_) :: w(j), x(j)
          REAL(rp_) :: zero, one, two
          PARAMETER (zero=0.0_rp_, one=1.0_rp_, two=2.0_rp_)
          REAL(rp_) :: half, four
          PARAMETER (half=0.5_rp_, four=4.0_rp_)
          REAL(rp_) :: absalp, absest, absgam, alpha, b, cosine, eps,       &
            norma, s1, s2, sine, t, test, tmp, zeta1, zeta2
          INTRINSIC :: ABS, MAX, SIGN, SQRT
          REAL(rp_) :: DDOT, DLAMCH
          EXTERNAL :: DDOT, DLAMCH
          eps = DLAMCH('Epsilon')
          alpha = DDOT(j, x, 1_ip_, w, 1_ip_)
          absalp = ABS(alpha)
          absgam = ABS(gamma)
          absest = ABS(sest)
          IF (job==1) THEN
            IF (sest==zero) THEN
              s1 = MAX(absgam, absalp)
              IF (s1==zero) THEN
                s = zero
                c = one
                sestpr = zero
              ELSE
                s = alpha/s1
                c = gamma/s1
                tmp = SQRT(s*s+c*c)
                s = s/tmp
                c = c/tmp
                sestpr = s1*tmp
              END IF
              RETURN
            ELSE IF (absgam<=eps*absest) THEN
              s = one
              c = zero
              tmp = MAX(absest, absalp)
              s1 = absest/tmp
              s2 = absalp/tmp
              sestpr = tmp*SQRT(s1*s1+s2*s2)
              RETURN
            ELSE IF (absalp<=eps*absest) THEN
              s1 = absgam
              s2 = absest
              IF (s1<=s2) THEN
                s = one
                c = zero
                sestpr = s2
              ELSE
                s = zero
                c = one
                sestpr = s1
              END IF
              RETURN
            ELSE IF (absest<=eps*absalp .OR. absest<=eps*absgam) THEN
              s1 = absgam
              s2 = absalp
              IF (s1<=s2) THEN
                tmp = s1/s2
                s = SQRT(one+tmp*tmp)
                sestpr = s2*s
                c = (gamma/s2)/s
                s = SIGN(one, alpha)/s
              ELSE
                tmp = s2/s1
                c = SQRT(one+tmp*tmp)
                sestpr = s1*c
                s = (alpha/s1)/c
                c = SIGN(one, gamma)/c
              END IF
              RETURN
            ELSE
              zeta1 = alpha/absest
              zeta2 = gamma/absest
              b = (one-zeta1*zeta1-zeta2*zeta2)*half
              c = zeta1*zeta1
              IF (b>zero) THEN
                t = c/(b+SQRT(b*b+c))
              ELSE
                t = SQRT(b*b+c) - b
              END IF
              sine = -zeta1/t
              cosine = -zeta2/(one+t)
              tmp = SQRT(sine*sine+cosine*cosine)
              s = sine/tmp
              c = cosine/tmp
              sestpr = SQRT(t+one)*absest
              RETURN
            END IF
          ELSE IF (job==2) THEN
            IF (sest==zero) THEN
              sestpr = zero
              IF (MAX(absgam,absalp)==zero) THEN
                sine = one
                cosine = zero
              ELSE
                sine = -gamma
                cosine = alpha
              END IF
              s1 = MAX(ABS(sine), ABS(cosine))
              s = sine/s1
              c = cosine/s1
              tmp = SQRT(s*s+c*c)
              s = s/tmp
              c = c/tmp
              RETURN
            ELSE IF (absgam<=eps*absest) THEN
              s = zero
              c = one
              sestpr = absgam
              RETURN
            ELSE IF (absalp<=eps*absest) THEN
              s1 = absgam
              s2 = absest
              IF (s1<=s2) THEN
                s = zero
                c = one
                sestpr = s1
              ELSE
                s = one
                c = zero
                sestpr = s2
              END IF
              RETURN
            ELSE IF (absest<=eps*absalp .OR. absest<=eps*absgam) THEN
              s1 = absgam
              s2 = absalp
              IF (s1<=s2) THEN
                tmp = s1/s2
                c = SQRT(one+tmp*tmp)
                sestpr = absest*(tmp/c)
                s = -(gamma/s2)/c
                c = SIGN(one, alpha)/c
              ELSE
                tmp = s2/s1
                s = SQRT(one+tmp*tmp)
                sestpr = absest/s
                c = (alpha/s1)/s
                s = -SIGN(one, gamma)/s
              END IF
              RETURN
            ELSE
              zeta1 = alpha/absest
              zeta2 = gamma/absest
              norma = MAX(one+zeta1*zeta1+ABS(zeta1*zeta2),                 &
                ABS(zeta1*zeta2)+zeta2*zeta2)
              test = one + two*(zeta1-zeta2)*(zeta1+zeta2)
              IF (test>=zero) THEN
                b = (zeta1*zeta1+zeta2*zeta2+one)*half
                c = zeta2*zeta2
                t = c/(b+SQRT(ABS(b*b-c)))
                sine = zeta1/(one-t)
                cosine = -zeta2/t
                sestpr = SQRT(t+four*eps*eps*norma)*absest
              ELSE
                b = (zeta2*zeta2+zeta1*zeta1-one)*half
                c = zeta1*zeta1
                IF (b>=zero) THEN
                  t = -c/(b+SQRT(b*b+c))
                ELSE
                  t = b - SQRT(b*b+c)
                END IF
                sine = -zeta1/t
                cosine = -zeta2/(one+t)
                sestpr = SQRT(one+t+four*eps*eps*norma)*absest
              END IF
              tmp = SQRT(sine*sine+cosine*cosine)
              s = sine/tmp
              c = cosine/tmp
              RETURN
            END IF
          END IF
          RETURN
        END SUBROUTINE

        LOGICAL FUNCTION DLAISNAN(din1, din2)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_), INTENT (IN) :: din1, din2
          DLAISNAN = (din1/=din2)
          RETURN
        END FUNCTION

        SUBROUTINE DLALS0(icompq, nl, nr, sqre, nrhs, b, ldb, bx, ldbx,     &
          perm, givptr, givcol, ldgcol, givnum, ldgnum, poles, difl, difr, z,&
          k, c, s, work, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: givptr, icompq, info, k, ldb, ldbx, ldgcol,       &
            ldgnum, nl, nr, nrhs, sqre
          REAL(rp_) :: c, s
          INTEGER(ip_) :: givcol(ldgcol, *), perm(*)
          REAL(rp_) :: b(ldb, *), bx(ldbx, *), difl(*), difr(ldgnum, *),    &
            givnum(ldgnum, *), poles(ldgnum, *), work(*), z(*)
          REAL(rp_) :: one, zero, negone
          PARAMETER (one=1.0_rp_, zero=0.0_rp_, negone=-1.0_rp_)
          INTEGER(ip_) :: i, j, m, n, nlp1
          REAL(rp_) :: diflj, difrj, dj, dsigj, dsigjp, temp
          EXTERNAL :: DCOPY, DGEMV, DLACPY, DLASCL, DROT, &
            DSCAL, XERBLA2
          REAL(rp_) :: DLAMC3, DNRM2
          EXTERNAL :: DLAMC3, DNRM2
          INTRINSIC :: MAX
          info = 0
          n = nl + nr + 1
          IF ((icompq<0) .OR. (icompq>1)) THEN
            info = -1
          ELSE IF (nl<1) THEN
            info = -2
          ELSE IF (nr<1) THEN
            info = -3
          ELSE IF ((sqre<0) .OR. (sqre>1)) THEN
            info = -4
          ELSE IF (nrhs<1) THEN
            info = -5
          ELSE IF (ldb<n) THEN
            info = -7
          ELSE IF (ldbx<n) THEN
            info = -9
          ELSE IF (givptr<0) THEN
            info = -11
          ELSE IF (ldgcol<n) THEN
            info = -13
          ELSE IF (ldgnum<n) THEN
            info = -15
          ELSE IF (k<1) THEN
            info = -20
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('LALS0', -info)
            RETURN
          END IF
          m = n + sqre
          nlp1 = nl + 1
          IF (icompq==0) THEN
            DO i = 1, givptr
              CALL DROT(nrhs, b(givcol(i,2),1), ldb, b(givcol(i, 1_ip_),1), &
                ldb, givnum(i,2), givnum(i,1))
            END DO
            CALL DCOPY(nrhs, b(nlp1,1), ldb, bx(1,1), ldbx)
            DO i = 2, n
              CALL DCOPY(nrhs, b(perm(i),1), ldb, bx(i,1), ldbx)
            END DO
            IF (k==1) THEN
              CALL DCOPY(nrhs, bx, ldbx, b, ldb)
              IF (z(1)<zero) THEN
                CALL DSCAL(nrhs, negone, b, ldb)
              END IF
            ELSE
              DO j = 1, k
                diflj = difl(j)
                dj = poles(j, 1_ip_)
                dsigj = -poles(j, 2_ip_)
                IF (j<k) THEN
                  difrj = -difr(j, 1_ip_)
                  dsigjp = -poles(j+1, 2_ip_)
                END IF
                IF ((z(j)==zero) .OR. (poles(j,2)==zero)) THEN
                  work(j) = zero
                ELSE
                  work(j) = -poles(j, 2_ip_)*z(j)/diflj/(poles(j,2)+dj)
                END IF
                DO i = 1, j - 1
                  IF ((z(i)==zero) .OR. (poles(i,2)==zero)) THEN
                    work(i) = zero
                  ELSE
                    work(i) = poles(i, 2_ip_)*z(i)/(DLAMC3(poles(i, 2_ip_), &
                      dsigj)-diflj)/(poles(i,2)+dj)
                  END IF
                END DO
                DO i = j + 1, k
                  IF ((z(i)==zero) .OR. (poles(i,2)==zero)) THEN
                    work(i) = zero
                  ELSE
                    work(i) = poles(i, 2_ip_)*z(i)/(DLAMC3(poles(i, 2_ip_), &
                      dsigjp)+difrj)/(poles(i,2)+dj)
                  END IF
                END DO
                work(1) = negone
                temp = DNRM2(k, work, 1_ip_)
                CALL DGEMV('T', k, nrhs, one, bx, ldbx, work, 1_ip_, zero,  &
                  b(j,1), ldb)
                CALL DLASCL('G', 0_ip_, 0_ip_, temp, one, 1_ip_, nrhs, b(j, &
                  1), ldb, info)
              END DO
            END IF
            IF (k<MAX(m,n)) CALL DLACPY('A', n-k, nrhs, bx(k+1,1), ldbx,    &
              b(k+1,1), ldb)
          ELSE
            IF (k==1) THEN
              CALL DCOPY(nrhs, b, ldb, bx, ldbx)
            ELSE
              DO j = 1, k
                dsigj = poles(j, 2_ip_)
                IF (z(j)==zero) THEN
                  work(j) = zero
                ELSE
                  work(j) = -z(j)/difl(j)/(dsigj+poles(j,1))/difr(j, 2_ip_)
                END IF
                DO i = 1, j - 1
                  IF (z(j)==zero) THEN
                    work(i) = zero
                  ELSE
                    work(i) = z(j)/(DLAMC3(dsigj,-poles(i+1,                &
                      2_ip_))-difr(i,1))/(dsigj+poles(i,1))/difr(i, 2_ip_)
                  END IF
                END DO
                DO i = j + 1, k
                  IF (z(j)==zero) THEN
                    work(i) = zero
                  ELSE
                    work(i) = z(j)/(DLAMC3(dsigj,-poles(i,                  &
                      2_ip_))-difl(i))/(dsigj+poles(i,1))/difr(i, 2_ip_)
                  END IF
                END DO
                CALL DGEMV('T', k, nrhs, one, b, ldb, work, 1_ip_, zero,    &
                  bx(j,1), ldbx)
              END DO
            END IF
            IF (sqre==1) THEN
              CALL DCOPY(nrhs, b(m,1), ldb, bx(m,1), ldbx)
              CALL DROT(nrhs, bx(1,1), ldbx, bx(m,1), ldbx, c, s)
            END IF
            IF (k<MAX(m,n)) CALL DLACPY('A', n-k, nrhs, b(k+1,1), ldb,      &
              bx(k+1,1), ldbx)
            CALL DCOPY(nrhs, bx(1,1), ldbx, b(nlp1,1), ldb)
            IF (sqre==1) THEN
              CALL DCOPY(nrhs, bx(m,1), ldbx, b(m,1), ldb)
            END IF
            DO i = 2, n
              CALL DCOPY(nrhs, bx(i,1), ldbx, b(perm(i),1), ldb)
            END DO
            DO i = givptr, 1_ip_, -1_ip_
              CALL DROT(nrhs, b(givcol(i,2),1), ldb, b(givcol(i, 1_ip_),1), &
                ldb, givnum(i,2), -givnum(i,1))
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLALSA(icompq, smlsiz, n, nrhs, b, ldb, bx, ldbx, u,     &
          ldu, vt, k, difl, difr, z, poles, givptr, givcol, ldgcol, perm,   &
          givnum, c, s, work, iwork, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: icompq, info, ldb, ldbx, ldgcol, ldu, n, nrhs,    &
            smlsiz
          INTEGER(ip_) :: givcol(ldgcol, *), givptr(*), iwork(*), k(*),     &
            perm(ldgcol, *)
          REAL(rp_) :: b(ldb, *), bx(ldbx, *), c(*), difl(ldu, *),          &
            difr(ldu, *), givnum(ldu, *), poles(ldu, *), s(*), u(ldu, *),   &
            vt(ldu, *), work(*), z(ldu, *)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          INTEGER(ip_) :: i, i1, ic, im1, inode, j, lf, ll, lvl, lvl2, nd,  &
            ndb1, ndiml, ndimr, nl, nlf, nlp1, nlvl, nr, nrf, nrp1, sqre
          EXTERNAL :: DCOPY, DGEMM, DLALS0, DLASDT, XERBLA2
          info = 0
          IF ((icompq<0) .OR. (icompq>1)) THEN
            info = -1
          ELSE IF (smlsiz<3) THEN
            info = -2
          ELSE IF (n<smlsiz) THEN
            info = -3
          ELSE IF (nrhs<1) THEN
            info = -4
          ELSE IF (ldb<n) THEN
            info = -6
          ELSE IF (ldbx<n) THEN
            info = -8
          ELSE IF (ldu<n) THEN
            info = -10
          ELSE IF (ldgcol<n) THEN
            info = -19
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('LALSA', -info)
            RETURN
          END IF
          inode = 1
          ndiml = inode + n
          ndimr = ndiml + n
          CALL DLASDT(n, nlvl, nd, iwork(inode), iwork(ndiml),              &
            iwork(ndimr), smlsiz)
          IF (icompq==1) THEN
            GO TO 50
          END IF
          ndb1 = (nd+1)/2
          DO i = ndb1, nd
            i1 = i - 1
            ic = iwork(inode+i1)
            nl = iwork(ndiml+i1)
            nr = iwork(ndimr+i1)
            nlf = ic - nl
            nrf = ic + 1
            CALL DGEMM('T', 'N', nl, nrhs, nl, one, u(nlf,1), ldu, b(nlf,   &
              1), ldb, zero, bx(nlf,1), ldbx)
            CALL DGEMM('T', 'N', nr, nrhs, nr, one, u(nrf,1), ldu, b(nrf,   &
              1), ldb, zero, bx(nrf,1), ldbx)
          END DO
          DO i = 1, nd
            ic = iwork(inode+i-1)
            CALL DCOPY(nrhs, b(ic,1), ldb, bx(ic,1), ldbx)
          END DO
          j = 2**nlvl
          sqre = 0
          DO lvl = nlvl, 1_ip_, -1_ip_
            lvl2 = 2*lvl - 1
            IF (lvl==1) THEN
              lf = 1
              ll = 1
            ELSE
              lf = 2**(lvl-1)
              ll = 2*lf - 1
            END IF
            DO i = lf, ll
              im1 = i - 1
              ic = iwork(inode+im1)
              nl = iwork(ndiml+im1)
              nr = iwork(ndimr+im1)
              nlf = ic - nl
              nrf = ic + 1
              j = j - 1
              CALL DLALS0(icompq, nl, nr, sqre, nrhs, bx(nlf,1), ldbx,      &
                b(nlf,1), ldb, perm(nlf,lvl), givptr(j), givcol(nlf,lvl2),  &
                ldgcol, givnum(nlf,lvl2), ldu, poles(nlf,lvl2), difl(nlf,   &
                lvl), difr(nlf,lvl2), z(nlf,lvl), k(j), c(j), s(j), work,   &
                info)
            END DO
          END DO
          GO TO 90
 50       CONTINUE
          j = 0
          DO lvl = 1, nlvl
            lvl2 = 2*lvl - 1
            IF (lvl==1) THEN
              lf = 1
              ll = 1
            ELSE
              lf = 2**(lvl-1)
              ll = 2*lf - 1
            END IF
            DO i = ll, lf, -1_ip_
              im1 = i - 1
              ic = iwork(inode+im1)
              nl = iwork(ndiml+im1)
              nr = iwork(ndimr+im1)
              nlf = ic - nl
              nrf = ic + 1
              IF (i==ll) THEN
                sqre = 0
              ELSE
                sqre = 1
              END IF
              j = j + 1
              CALL DLALS0(icompq, nl, nr, sqre, nrhs, b(nlf,1), ldb,        &
                bx(nlf,1), ldbx, perm(nlf,lvl), givptr(j), givcol(nlf,lvl2),&
                ldgcol, givnum(nlf,lvl2), ldu, poles(nlf,lvl2), difl(nlf,   &
                lvl), difr(nlf,lvl2), z(nlf,lvl), k(j), c(j), s(j), work,   &
                info)
            END DO
          END DO
          ndb1 = (nd+1)/2
          DO i = ndb1, nd
            i1 = i - 1
            ic = iwork(inode+i1)
            nl = iwork(ndiml+i1)
            nr = iwork(ndimr+i1)
            nlp1 = nl + 1
            IF (i==nd) THEN
              nrp1 = nr
            ELSE
              nrp1 = nr + 1
            END IF
            nlf = ic - nl
            nrf = ic + 1
            CALL DGEMM('T', 'N', nlp1, nrhs, nlp1, one, vt(nlf,1), ldu,     &
              b(nlf,1), ldb, zero, bx(nlf,1), ldbx)
            CALL DGEMM('T', 'N', nrp1, nrhs, nrp1, one, vt(nrf,1), ldu,     &
              b(nrf,1), ldb, zero, bx(nrf,1), ldbx)
          END DO
 90       CONTINUE
          RETURN
        END SUBROUTINE

        SUBROUTINE DLALSD(uplo, smlsiz, n, nrhs, d, e, b, ldb, rcond, rank, &
          work, iwork, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, ldb, n, nrhs, rank, smlsiz
          REAL(rp_) :: rcond
          INTEGER(ip_) :: iwork(*)
          REAL(rp_) :: b(ldb, *), d(*), e(*), work(*)
          REAL(rp_) :: zero, one, two
          PARAMETER (zero=0.0_rp_, one=1.0_rp_, two=2.0_rp_)
          INTEGER(ip_) :: bx, bxst, c, difl, difr, givcol, givnum, givptr,  &
            i, icmpq1, icmpq2, iwk, j, k, nlvl, nm1, nsize, nsub, nwork,    &
            perm, poles, s, sizei, smlszp, sqre, st, st1, u, vt, z
          REAL(rp_) :: cs, eps, orgnrm, r, rcnd, sn, tol
          INTEGER(ip_) :: IDAMAX
          REAL(rp_) :: DLAMCH, DLANST
          EXTERNAL :: IDAMAX, DLAMCH, DLANST
          EXTERNAL :: DCOPY, DGEMM, DLACPY, DLALSA, DLARTG,& 
            DLASCL, DLASDA, DLASDQ, DLASET, DLASRT, DROT, &
            XERBLA2
          INTRINSIC :: ABS, REAL, INT, LOG, SIGN
          info = 0
          IF (n<0) THEN
            info = -3
          ELSE IF (nrhs<1) THEN
            info = -4
          ELSE IF ((ldb<1) .OR. (ldb<n)) THEN
            info = -8
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('LALSD', -info)
            RETURN
          END IF
          eps = DLAMCH('Epsilon')
          IF ((rcond<=zero) .OR. (rcond>=one)) THEN
            rcnd = eps
          ELSE
            rcnd = rcond
          END IF
          rank = 0
          IF (n==0) THEN
            RETURN
          ELSE IF (n==1) THEN
            IF (d(1)==zero) THEN
              CALL DLASET('A', 1_ip_, nrhs, zero, zero, b, ldb)
            ELSE
              rank = 1
              CALL DLASCL('G', 0_ip_, 0_ip_, d(1), one, 1_ip_, nrhs, b,     &
                ldb, info)
              d(1) = ABS(d(1))
            END IF
            RETURN
          END IF
          IF (uplo=='L') THEN
            DO i = 1, n - 1
              CALL DLARTG(d(i), e(i), cs, sn, r)
              d(i) = r
              e(i) = sn*d(i+1)
              d(i+1) = cs*d(i+1)
              IF (nrhs==1) THEN
                CALL DROT(1_ip_, b(i,1), 1_ip_, b(i+1,1), 1_ip_, cs, sn)
              ELSE
                work(i*2-1) = cs
                work(i*2) = sn
              END IF
            END DO
            IF (nrhs>1) THEN
              DO i = 1, nrhs
                DO j = 1, n - 1
                  cs = work(j*2-1)
                  sn = work(j*2)
                  CALL DROT(1_ip_, b(j,i), 1_ip_, b(j+1,i), 1_ip_, cs, sn)
                END DO
              END DO
            END IF
          END IF
          nm1 = n - 1
          orgnrm = DLANST('M', n, d, e)
          IF (orgnrm==zero) THEN
            CALL DLASET('A', n, nrhs, zero, zero, b, ldb)
            RETURN
          END IF
          CALL DLASCL('G', 0_ip_, 0_ip_, orgnrm, one, n, 1_ip_, d, n, info)
          CALL DLASCL('G', 0_ip_, 0_ip_, orgnrm, one, nm1, 1_ip_, e, nm1,   &
            info)
          IF (n<=smlsiz) THEN
            nwork = 1 + n*n
            CALL DLASET('A', n, n, zero, one, work, n)
            CALL DLASDQ('U', 0_ip_, n, n, 0_ip_, nrhs, d, e, work, n, work, &
              n, b, ldb, work(nwork), info)
            IF (info/=0) THEN
              RETURN
            END IF
            tol = rcnd*ABS(d(IDAMAX(n,d, 1_ip_)))
            DO i = 1, n
              IF (d(i)<=tol) THEN
                CALL DLASET('A', 1_ip_, nrhs, zero, zero, b(i,1), ldb)
              ELSE
                CALL DLASCL('G', 0_ip_, 0_ip_, d(i), one, 1_ip_, nrhs, b(i, &
                  1), ldb, info)
                rank = rank + 1
              END IF
            END DO
            CALL DGEMM('T', 'N', n, nrhs, n, one, work, n, b, ldb, zero,    &
              work(nwork), n)
            CALL DLACPY('A', n, nrhs, work(nwork), n, b, ldb)
            CALL DLASCL('G', 0_ip_, 0_ip_, one, orgnrm, n, 1_ip_, d, n,     &
              info)
            CALL DLASRT('D', n, d, info)
            CALL DLASCL('G', 0_ip_, 0_ip_, orgnrm, one, n, nrhs, b, ldb,    &
              info)
            RETURN
          END IF
          nlvl = INT(LOG(REAL(n,rp_)/REAL(smlsiz+1,rp_))/LOG(two)) + 1
          smlszp = smlsiz + 1
          u = 1
          vt = 1 + smlsiz*n
          difl = vt + smlszp*n
          difr = difl + nlvl*n
          z = difr + nlvl*n*2
          c = z + nlvl*n
          s = c + n
          poles = s + n
          givnum = poles + 2*nlvl*n
          bx = givnum + 2*nlvl*n
          nwork = bx + n*nrhs
          sizei = 1 + n
          k = sizei + n
          givptr = k + n
          perm = givptr + n
          givcol = perm + nlvl*n
          iwk = givcol + nlvl*n*2
          st = 1
          sqre = 0
          icmpq1 = 1
          icmpq2 = 0
          nsub = 0
          DO i = 1, n
            IF (ABS(d(i))<eps) THEN
              d(i) = SIGN(eps, d(i))
            END IF
          END DO
          DO i = 1, nm1
            IF ((ABS(e(i))<eps) .OR. (i==nm1)) THEN
              nsub = nsub + 1
              iwork(nsub) = st
              IF (i<nm1) THEN
                nsize = i - st + 1
                iwork(sizei+nsub-1) = nsize
              ELSE IF (ABS(e(i))>=eps) THEN
                nsize = n - st + 1
                iwork(sizei+nsub-1) = nsize
              ELSE
                nsize = i - st + 1
                iwork(sizei+nsub-1) = nsize
                nsub = nsub + 1
                iwork(nsub) = n
                iwork(sizei+nsub-1) = 1
                CALL DCOPY(nrhs, b(n,1), ldb, work(bx+nm1), n)
              END IF
              st1 = st - 1
              IF (nsize==1) THEN
                CALL DCOPY(nrhs, b(st,1), ldb, work(bx+st1), n)
              ELSE IF (nsize<=smlsiz) THEN
                CALL DLASET('A', nsize, nsize, zero, one, work(vt+st1), n)
                CALL DLASDQ('U', 0_ip_, nsize, nsize, 0_ip_, nrhs, d(st),   &
                  e(st), work(vt+st1), n, work(nwork), n, b(st,1), ldb,     &
                  work(nwork), info)
                IF (info/=0) THEN
                  RETURN
                END IF
                CALL DLACPY('A', nsize, nrhs, b(st,1), ldb, work(bx+st1),   &
                  n)
              ELSE
                CALL DLASDA(icmpq1, smlsiz, nsize, sqre, d(st), e(st),      &
                  work(u+st1), n, work(vt+st1), iwork(k+st1), work(difl+st1),&
                  work(difr+st1), work(z+st1), work(poles+st1),             &
                  iwork(givptr+st1), iwork(givcol+st1), n, iwork(perm+st1), &
                  work(givnum+st1), work(c+st1), work(s+st1), work(nwork),  &
                  iwork(iwk), info)
                IF (info/=0) THEN
                  RETURN
                END IF
                bxst = bx + st1
                CALL DLALSA(icmpq2, smlsiz, nsize, nrhs, b(st,1), ldb,      &
                  work(bxst), n, work(u+st1), n, work(vt+st1), iwork(k+st1),&
                  work(difl+st1), work(difr+st1), work(z+st1),              &
                  work(poles+st1), iwork(givptr+st1), iwork(givcol+st1), n, &
                  iwork(perm+st1), work(givnum+st1), work(c+st1),           &
                  work(s+st1), work(nwork), iwork(iwk), info)
                IF (info/=0) THEN
                  RETURN
                END IF
              END IF
              st = i + 1
            END IF
          END DO
          tol = rcnd*ABS(d(IDAMAX(n,d, 1_ip_)))
          DO i = 1, n
            IF (ABS(d(i))<=tol) THEN
              CALL DLASET('A', 1_ip_, nrhs, zero, zero, work(bx+i-1), n)
            ELSE
              rank = rank + 1
              CALL DLASCL('G', 0_ip_, 0_ip_, d(i), one, 1_ip_, nrhs,        &
                work(bx+i-1), n, info)
            END IF
            d(i) = ABS(d(i))
          END DO
          icmpq2 = 1
          DO i = 1, nsub
            st = iwork(i)
            st1 = st - 1
            nsize = iwork(sizei+i-1)
            bxst = bx + st1
            IF (nsize==1) THEN
              CALL DCOPY(nrhs, work(bxst), n, b(st,1), ldb)
            ELSE IF (nsize<=smlsiz) THEN
              CALL DGEMM('T', 'N', nsize, nrhs, nsize, one, work(vt+st1),   &
                n, work(bxst), n, zero, b(st,1), ldb)
            ELSE
              CALL DLALSA(icmpq2, smlsiz, nsize, nrhs, work(bxst), n, b(st, &
                1), ldb, work(u+st1), n, work(vt+st1), iwork(k+st1),        &
                work(difl+st1), work(difr+st1), work(z+st1), work(poles+st1),&
                iwork(givptr+st1), iwork(givcol+st1), n, iwork(perm+st1),   &
                work(givnum+st1), work(c+st1), work(s+st1), work(nwork),    &
                iwork(iwk), info)
              IF (info/=0) THEN
                RETURN
              END IF
            END IF
          END DO
          CALL DLASCL('G', 0_ip_, 0_ip_, one, orgnrm, n, 1_ip_, d, n, info)
          CALL DLASRT('D', n, d, info)
          CALL DLASCL('G', 0_ip_, 0_ip_, orgnrm, one, n, nrhs, b, ldb,      &
            info)
          RETURN
        END SUBROUTINE

        REAL(rp_) FUNCTION DLAMCH(cmach)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: cmach
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          REAL(rp_) :: rnd, eps, sfmin, small, rmach
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          INTRINSIC :: DIGITS, EPSILON, HUGE, MAXEXPONENT, MINEXPONENT,     &
            RADIX, TINY
          rnd = one
          IF (one==rnd) THEN
            eps = EPSILON(zero)*0.5
          ELSE
            eps = EPSILON(zero)
          END IF
          IF (LSAME(cmach,'E')) THEN
            rmach = eps
          ELSE IF (LSAME(cmach,'S')) THEN
            sfmin = TINY(zero)
            small = one/HUGE(zero)
            IF (small>=sfmin) THEN
              sfmin = small*(one+eps)
            END IF
            rmach = sfmin
          ELSE IF (LSAME(cmach,'B')) THEN
            rmach = RADIX(zero)
          ELSE IF (LSAME(cmach,'P')) THEN
            rmach = eps*RADIX(zero)
          ELSE IF (LSAME(cmach,'N')) THEN
            rmach = DIGITS(zero)
          ELSE IF (LSAME(cmach,'R')) THEN
            rmach = rnd
          ELSE IF (LSAME(cmach,'M')) THEN
            rmach = MINEXPONENT(zero)
          ELSE IF (LSAME(cmach,'U')) THEN
            rmach = TINY(zero)
          ELSE IF (LSAME(cmach,'L')) THEN
            rmach = MAXEXPONENT(zero)
          ELSE IF (LSAME(cmach,'O')) THEN
            rmach = HUGE(zero)
          ELSE
            rmach = zero
          END IF
          DLAMCH = rmach
          RETURN
        END FUNCTION

        REAL(rp_) FUNCTION DLAMC3(a, b)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: a, b
          DLAMC3 = a + b
          RETURN
        END FUNCTION

        SUBROUTINE DLAMRG(n1, n2, a, dtrd1, dtrd2, index)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: dtrd1, dtrd2, n1, n2
          INTEGER(ip_) :: index(*)
          REAL(rp_) :: a(*)
          INTEGER(ip_) :: i, ind1, ind2, n1sv, n2sv
          n1sv = n1
          n2sv = n2
          IF (dtrd1>0) THEN
            ind1 = 1
          ELSE
            ind1 = n1
          END IF
          IF (dtrd2>0) THEN
            ind2 = 1 + n1
          ELSE
            ind2 = n1 + n2
          END IF
          i = 1
 10       CONTINUE
          IF (n1sv>0 .AND. n2sv>0) THEN
            IF (a(ind1)<=a(ind2)) THEN
              index(i) = ind1
              i = i + 1
              ind1 = ind1 + dtrd1
              n1sv = n1sv - 1
            ELSE
              index(i) = ind2
              i = i + 1
              ind2 = ind2 + dtrd2
              n2sv = n2sv - 1
            END IF
            GO TO 10
          END IF
          IF (n1sv==0) THEN
            DO n1sv = 1, n2sv
              index(i) = ind2
              i = i + 1
              ind2 = ind2 + dtrd2
            END DO
          ELSE
            DO n2sv = 1, n1sv
              index(i) = ind1
              i = i + 1
              ind1 = ind1 + dtrd1
            END DO
          END IF
          RETURN
        END SUBROUTINE

        REAL(rp_) FUNCTION DLANGE(norm, m, n, a, lda, work)
          USE BLAS_LAPACK_KINDS_precision
          IMPLICIT NONE
          CHARACTER :: norm
          INTEGER(ip_) :: lda, m, n
          REAL(rp_) :: a(lda, *), work(*)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          INTEGER(ip_) :: i, j
          REAL(rp_) :: sum, value, temp
          REAL(rp_) :: ssq(2), colssq(2)
          LOGICAL :: LISNAN
          EXTERNAL :: DLASSQ, DCOMBSSQ
          LOGICAL :: LSAME, DISNAN
          EXTERNAL :: LSAME, DISNAN
          INTRINSIC :: ABS, MIN, SQRT
          IF (MIN(m,n)==0) THEN
            value = zero
          ELSE IF (LSAME(norm,'M')) THEN
            value = zero
            DO j = 1, n
              DO i = 1, m
                temp = ABS(a(i,j))
                LISNAN = DISNAN(temp)
                IF (value<temp .OR. LISNAN) value = temp
              END DO
            END DO
          ELSE IF ((LSAME(norm,'O')) .OR. (norm=='1')) THEN
            value = zero
            DO j = 1, n
              sum = zero
              DO i = 1, m
                sum = sum + ABS(a(i,j))
              END DO
              LISNAN = DISNAN(sum)
              IF (value<sum .OR. LISNAN) value = sum
            END DO
          ELSE IF (LSAME(norm,'I')) THEN
            DO i = 1, m
              work(i) = zero
            END DO
            DO j = 1, n
              DO i = 1, m
                work(i) = work(i) + ABS(a(i,j))
              END DO
            END DO
            value = zero
            DO i = 1, m
              temp = work(i)
              LISNAN = DISNAN(temp)
              IF (value<temp .OR. LISNAN) value = temp
            END DO
          ELSE IF ((LSAME(norm,'F')) .OR. (LSAME(norm,'E'))) THEN
            ssq(1) = zero
            ssq(2) = one
            DO j = 1, n
              colssq(1) = zero
              colssq(2) = one
              CALL DLASSQ(m, a(1,j), 1_ip_, colssq(1), colssq(2))
              CALL DCOMBSSQ(ssq, colssq)
            END DO
            value = ssq(1)*SQRT(ssq(2))
          END IF
          DLANGE = value
          RETURN
        END FUNCTION

        REAL(rp_) FUNCTION DLANST(norm, n, d, e)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: norm
          INTEGER(ip_) :: n
          REAL(rp_) :: d(*), e(*)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          INTEGER(ip_) :: i
          REAL(rp_) :: anorm, scale, sum
          LOGICAL :: LISNAN
          LOGICAL :: LSAME, DISNAN
          EXTERNAL :: LSAME, DISNAN
          EXTERNAL :: DLASSQ
          INTRINSIC :: ABS, SQRT
          IF (n<=0) THEN
            anorm = zero
          ELSE IF (LSAME(norm,'M')) THEN
            anorm = ABS(d(n))
            DO i = 1, n - 1
              sum = ABS(d(i))
              LISNAN = DISNAN(sum)
              IF (anorm<sum .OR. LISNAN) anorm = sum
              sum = ABS(e(i))
              LISNAN = DISNAN(sum)
              IF (anorm<sum .OR. LISNAN) anorm = sum
            END DO
          ELSE IF (LSAME(norm,'O') .OR. norm=='1' .OR. LSAME(norm,& 
            'I')) THEN
            IF (n==1) THEN
              anorm = ABS(d(1))
            ELSE
              anorm = ABS(d(1)) + ABS(e(1))
              sum = ABS(e(n-1)) + ABS(d(n))
              LISNAN = DISNAN(sum)
              IF (anorm<sum .OR. LISNAN) anorm = sum
              DO i = 2, n - 1
                sum = ABS(d(i)) + ABS(e(i)) + ABS(e(i-1))
                LISNAN = DISNAN(sum)
                IF (anorm<sum .OR. LISNAN) anorm = sum
              END DO
            END IF
          ELSE IF ((LSAME(norm,'F')) .OR. (LSAME(norm,'E'))) THEN
            scale = zero
            sum = one
            IF (n>1) THEN
              CALL DLASSQ(n-1, e, 1_ip_, scale, sum)
              sum = 2*sum
            END IF
            CALL DLASSQ(n, d, 1_ip_, scale, sum)
            anorm = scale*SQRT(sum)
          END IF
          DLANST = anorm
          RETURN
        END FUNCTION

        REAL(rp_) FUNCTION DLANSY(norm, uplo, n, a, lda, work)
          USE BLAS_LAPACK_KINDS_precision
          IMPLICIT NONE
          CHARACTER :: norm, uplo
          INTEGER(ip_) :: lda, n
          REAL(rp_) :: a(lda, *), work(*)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          INTEGER(ip_) :: i, j
          REAL(rp_) :: absa, sum, value
          REAL(rp_) :: ssq(2), colssq(2)
          LOGICAL :: LISNAN
          LOGICAL :: LSAME, DISNAN
          EXTERNAL :: LSAME, DISNAN
          EXTERNAL :: DLASSQ, DCOMBSSQ
          INTRINSIC :: ABS, SQRT
          IF (n==0) THEN
            value = zero
          ELSE IF (LSAME(norm,'M')) THEN
            value = zero
            IF (LSAME(uplo,'U')) THEN
              DO j = 1, n
                DO i = 1, j
                  sum = ABS(a(i,j))
                  LISNAN = DISNAN(sum)
                  IF (value<sum .OR. LISNAN) value = sum
                END DO
              END DO
            ELSE
              DO j = 1, n
                DO i = j, n
                  sum = ABS(a(i,j))
                  LISNAN = DISNAN(sum)
                  IF (value<sum .OR. LISNAN) value = sum
                END DO
              END DO
            END IF
          ELSE IF ((LSAME(norm,'I')) .OR. (LSAME(norm, 'O')) .OR. &
            (norm=='1')) THEN
            value = zero
            IF (LSAME(uplo,'U')) THEN
              DO j = 1, n
                sum = zero
                DO i = 1, j - 1
                  absa = ABS(a(i,j))
                  sum = sum + absa
                  work(i) = work(i) + absa
                END DO
                work(j) = sum + ABS(a(j,j))
              END DO
              DO i = 1, n
                sum = work(i)
                LISNAN = DISNAN(sum)
                IF (value<sum .OR. LISNAN) value = sum
              END DO
            ELSE
              DO i = 1, n
                work(i) = zero
              END DO
              DO j = 1, n
                sum = work(j) + ABS(a(j,j))
                DO i = j + 1, n
                  absa = ABS(a(i,j))
                  sum = sum + absa
                  work(i) = work(i) + absa
                END DO
                LISNAN = DISNAN(sum)
                IF (value<sum .OR. LISNAN) value = sum
              END DO
            END IF
          ELSE IF ((LSAME(norm,'F')) .OR. (LSAME(norm,'E'))) THEN
            ssq(1) = zero
            ssq(2) = one
            IF (LSAME(uplo,'U')) THEN
              DO j = 2, n
                colssq(1) = zero
                colssq(2) = one
                CALL DLASSQ(j-1, a(1,j), 1_ip_, colssq(1), colssq(2))
                CALL DCOMBSSQ(ssq, colssq)
              END DO
            ELSE
              DO j = 1, n - 1
                colssq(1) = zero
                colssq(2) = one
                CALL DLASSQ(n-j, a(j+1,j), 1_ip_, colssq(1), colssq(2))
                CALL DCOMBSSQ(ssq, colssq)
              END DO
            END IF
            ssq(2) = 2*ssq(2)
            colssq(1) = zero
            colssq(2) = one
            CALL DLASSQ(n, a, lda+1, colssq(1), colssq(2))
            CALL DCOMBSSQ(ssq, colssq)
            value = ssq(1)*SQRT(ssq(2))
          END IF
          DLANSY = value
          RETURN
        END FUNCTION

        SUBROUTINE DLANV2(a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: a, b, c, cs, d, rt1i, rt1r, rt2i, rt2r, sn
          REAL(rp_) :: zero, half, one, two
          PARAMETER (zero=0.0_rp_, half=0.5_rp_, one=1.0_rp_, two=2.0_rp_)
          REAL(rp_) :: multpl
          PARAMETER (multpl=4.0_rp_)
          REAL(rp_) :: aa, bb, bcmax, bcmis, cc, cs1, dd, eps, p, sab, sac, &
            scale, sigma, sn1, tau, temp, z, safmin, safmn2, safmx2
          INTEGER(ip_) :: count
          REAL(rp_) :: DLAMCH, DLAPY2
          EXTERNAL :: DLAMCH, DLAPY2
          INTRINSIC :: ABS, MAX, MIN, SIGN, SQRT
          safmin = DLAMCH('S')
          eps = DLAMCH('P')
          safmn2 = DLAMCH('B')**INT(LOG(safmin/eps)/LOG(DLAMCH('B'))/two)
          safmx2 = one/safmn2
          IF (c==zero) THEN
            cs = one
            sn = zero
          ELSE IF (b==zero) THEN
            cs = zero
            sn = one
            temp = d
            d = a
            a = temp
            b = -c
            c = zero
          ELSE IF ((a-d)==zero .AND. SIGN(one,b)/=SIGN(one,c)) THEN
            cs = one
            sn = zero
          ELSE
            temp = a - d
            p = half*temp
            bcmax = MAX(ABS(b), ABS(c))
            bcmis = MIN(ABS(b), ABS(c))*SIGN(one, b)*SIGN(one, c)
            scale = MAX(ABS(p), bcmax)
            z = (p/scale)*p + (bcmax/scale)*bcmis
            IF (z>=multpl*eps) THEN
              z = p + SIGN(SQRT(scale)*SQRT(z), p)
              a = d + z
              d = d - (bcmax/z)*bcmis
              tau = DLAPY2(c, z)
              cs = z/tau
              sn = c/tau
              b = b - c
              c = zero
            ELSE
              count = 0
              sigma = b + c
 10           CONTINUE
              count = count + 1
              scale = MAX(ABS(temp), ABS(sigma))
              IF (scale>=safmx2) THEN
                sigma = sigma*safmn2
                temp = temp*safmn2
                IF (count<=20) GO TO 10
              END IF
              IF (scale<=safmn2) THEN
                sigma = sigma*safmx2
                temp = temp*safmx2
                IF (count<=20) GO TO 10
              END IF
              p = half*temp
              tau = DLAPY2(sigma, temp)
              cs = SQRT(half*(one+ABS(sigma)/tau))
              sn = -(p/(tau*cs))*SIGN(one, sigma)
              aa = a*cs + b*sn
              bb = -a*sn + b*cs
              cc = c*cs + d*sn
              dd = -c*sn + d*cs
              a = aa*cs + cc*sn
              b = bb*cs + dd*sn
              c = -aa*sn + cc*cs
              d = -bb*sn + dd*cs
              temp = half*(a+d)
              a = temp
              d = temp
              IF (c/=zero) THEN
                IF (b/=zero) THEN
                  IF (SIGN(one,b)==SIGN(one,c)) THEN
                    sab = SQRT(ABS(b))
                    sac = SQRT(ABS(c))
                    p = SIGN(sab*sac, c)
                    tau = one/SQRT(ABS(b+c))
                    a = temp + p
                    d = temp - p
                    b = b - c
                    c = zero
                    cs1 = sab*tau
                    sn1 = sac*tau
                    temp = cs*cs1 - sn*sn1
                    sn = cs*sn1 + sn*cs1
                    cs = temp
                  END IF
                ELSE
                  b = -c
                  c = zero
                  temp = cs
                  cs = -sn
                  sn = temp
                END IF
              END IF
            END IF
          END IF
          rt1r = a
          rt2r = d
          IF (c==zero) THEN
            rt1i = zero
            rt2i = zero
          ELSE
            rt1i = SQRT(ABS(b))*SQRT(ABS(c))
            rt2i = -rt1i
          END IF
          RETURN
        END SUBROUTINE

        REAL(rp_) FUNCTION DLAPY2(x, y)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: x, y
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          REAL(rp_) :: w, xabs, yabs, z
          LOGICAL :: x_is_nan, y_is_nan
          LOGICAL :: DISNAN
          EXTERNAL :: DISNAN
          INTRINSIC :: ABS, MAX, MIN, SQRT
          x_is_nan = DISNAN(x)
          y_is_nan = DISNAN(y)
          IF (x_is_nan) DLAPY2 = x
          IF (y_is_nan) DLAPY2 = y
          IF (.NOT. (x_is_nan .OR. y_is_nan)) THEN
            xabs = ABS(x)
            yabs = ABS(y)
            w = MAX(xabs, yabs)
            z = MIN(xabs, yabs)
            IF (z==zero) THEN
              DLAPY2 = w
            ELSE
              DLAPY2 = w*SQRT(one+(z/w)**2)
            END IF
          END IF
          RETURN
        END FUNCTION

        SUBROUTINE DLAQP2(m, n, offset, a, lda, jpvt, tau, vn1, vn2, work)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: lda, m, n, offset
          INTEGER(ip_) :: jpvt(*)
          REAL(rp_) :: a(lda, *), tau(*), vn1(*), vn2(*), work(*)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          INTEGER(ip_) :: i, itemp, j, mn, offpi, pvt
          REAL(rp_) :: aii, temp, temp2, tol3z
          EXTERNAL :: DLARF, DLARFG, DSWAP
          INTRINSIC :: ABS, MAX, MIN, SQRT
          INTEGER(ip_) :: IDAMAX
          REAL(rp_) :: DLAMCH, DNRM2
          EXTERNAL :: IDAMAX, DLAMCH, DNRM2
          mn = MIN(m-offset, n)
          tol3z = SQRT(DLAMCH('Epsilon'))
          DO i = 1, mn
            offpi = offset + i
            pvt = (i-1) + IDAMAX(n-i+1, vn1(i), 1_ip_)
            IF (pvt/=i) THEN
              CALL DSWAP(m, a(1,pvt), 1_ip_, a(1,i), 1_ip_)
              itemp = jpvt(pvt)
              jpvt(pvt) = jpvt(i)
              jpvt(i) = itemp
              vn1(pvt) = vn1(i)
              vn2(pvt) = vn2(i)
            END IF
            IF (offpi<m) THEN
              CALL DLARFG(m-offpi+1, a(offpi,i), a(offpi+1,i), 1_ip_,       &
                tau(i))
            ELSE
              CALL DLARFG(1_ip_, a(m,i), a(m,i), 1_ip_, tau(i))
            END IF
            IF (i<n) THEN
              aii = a(offpi, i)
              a(offpi, i) = one
              CALL DLARF('Left', m-offpi+1, n-i, a(offpi,i), 1_ip_, tau(i), &
                a(offpi,i+1), lda, work(1))
              a(offpi, i) = aii
            END IF
            DO j = i + 1, n
              IF (vn1(j)/=zero) THEN
                temp = one - (ABS(a(offpi,j))/vn1(j))**2
                temp = MAX(temp, zero)
                temp2 = temp*(vn1(j)/vn2(j))**2
                IF (temp2<=tol3z) THEN
                  IF (offpi<m) THEN
                    vn1(j) = DNRM2(m-offpi, a(offpi+1,j), 1_ip_)
                    vn2(j) = vn1(j)
                  ELSE
                    vn1(j) = zero
                    vn2(j) = zero
                  END IF
                ELSE
                  vn1(j) = vn1(j)*SQRT(temp)
                END IF
              END IF
            END DO
          END DO
          RETURN
        END SUBROUTINE

        SUBROUTINE DLAQPS(m, n, offset, nb, kb, a, lda, jpvt, tau, vn1,     &
          vn2, auxv, f, ldf)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: kb, lda, ldf, m, n, nb, offset
          INTEGER(ip_) :: jpvt(*)
          REAL(rp_) :: a(lda, *), auxv(*), f(ldf, *), tau(*), vn1(*),       &
            vn2(*)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          INTEGER(ip_) :: itemp, j, k, lastrk, lsticc, pvt, rk
          REAL(rp_) :: akk, temp, temp2, tol3z
          EXTERNAL :: DGEMM, DGEMV, DLARFG, DSWAP
          INTRINSIC :: ABS, REAL, MAX, MIN, NINT, SQRT
          INTEGER(ip_) :: IDAMAX
          REAL(rp_) :: DLAMCH, DNRM2
          EXTERNAL :: IDAMAX, DLAMCH, DNRM2
          lastrk = MIN(m, n+offset)
          lsticc = 0
          k = 0
          tol3z = SQRT(DLAMCH('Epsilon'))
 10       CONTINUE
          IF ((k<nb) .AND. (lsticc==0)) THEN
            k = k + 1
            rk = offset + k
            pvt = (k-1) + IDAMAX(n-k+1, vn1(k), 1_ip_)
            IF (pvt/=k) THEN
              CALL DSWAP(m, a(1,pvt), 1_ip_, a(1,k), 1_ip_)
              CALL DSWAP(k-1, f(pvt,1), ldf, f(k,1), ldf)
              itemp = jpvt(pvt)
              jpvt(pvt) = jpvt(k)
              jpvt(k) = itemp
              vn1(pvt) = vn1(k)
              vn2(pvt) = vn2(k)
            END IF
            IF (k>1) THEN
              CALL DGEMV('No transpose', m-rk+1, k-1, -one, a(rk,1), lda,   &
                f(k,1), ldf, one, a(rk,k), 1_ip_)
            END IF
            IF (rk<m) THEN
              CALL DLARFG(m-rk+1, a(rk,k), a(rk+1,k), 1_ip_, tau(k))
            ELSE
              CALL DLARFG(1_ip_, a(rk,k), a(rk,k), 1_ip_, tau(k))
            END IF
            akk = a(rk, k)
            a(rk, k) = one
            IF (k<n) THEN
              CALL DGEMV('Transpose', m-rk+1, n-k, tau(k), a(rk,k+1), lda,  &
                a(rk,k), 1_ip_, zero, f(k+1,k), 1_ip_)
            END IF
            DO j = 1, k
              f(j, k) = zero
            END DO
            IF (k>1) THEN
              CALL DGEMV('Transpose', m-rk+1, k-1, -tau(k), a(rk,1), lda,   &
                a(rk,k), 1_ip_, zero, auxv(1), 1_ip_)
              CALL DGEMV('No transpose', n, k-1, one, f(1,1), ldf, auxv(1), &
                1_ip_, one, f(1,k), 1_ip_)
            END IF
            IF (k<n) THEN
              CALL DGEMV('No transpose', n-k, k, -one, f(k+1,1), ldf, a(rk, &
                1), lda, one, a(rk,k+1), lda)
            END IF
            IF (rk<lastrk) THEN
              DO j = k + 1, n
                IF (vn1(j)/=zero) THEN
                  temp = ABS(a(rk,j))/vn1(j)
                  temp = MAX(zero, (one+temp)*(one-temp))
                  temp2 = temp*(vn1(j)/vn2(j))**2
                  IF (temp2<=tol3z) THEN
                    vn2(j) = REAL(lsticc,rp_)
                    lsticc = j
                  ELSE
                    vn1(j) = vn1(j)*SQRT(temp)
                  END IF
                END IF
              END DO
            END IF
            a(rk, k) = akk
            GO TO 10
          END IF
          kb = k
          rk = offset + kb
          IF (kb<MIN(n,m-offset)) THEN
            CALL DGEMM('No transpose', 'Transpose', m-rk, n-kb, kb, -one,   &
              a(rk+1,1), lda, f(kb+1,1), ldf, one, a(rk+1,kb+1), lda)
          END IF
 40       CONTINUE
          IF (lsticc>0) THEN
            itemp = NINT(vn2(lsticc))
            vn1(lsticc) = DNRM2(m-rk, a(rk+1,lsticc), 1_ip_)
            vn2(lsticc) = vn1(lsticc)
            lsticc = itemp
            GO TO 40
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLAQR0(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, iloz,  &
          ihiz, z, ldz, work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: ihi, ihiz, ilo, iloz, info, ldh, ldz, lwork, n
          LOGICAL :: wantt, wantz
          REAL(rp_) :: h(ldh, *), wi(*), work(*), wr(*), z(ldz, *)
          INTEGER(ip_) :: ntiny
          PARAMETER (ntiny=15)
          INTEGER(ip_) :: kexnw
          PARAMETER (kexnw=5)
          INTEGER(ip_) :: kexsh
          PARAMETER (kexsh=6)
          REAL(rp_) :: wilk1, wilk2
          PARAMETER (wilk1=0.75_rp_, wilk2=-0.4375_rp_)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          REAL(rp_) :: aa, bb, cc, cs, dd, sn, ss, swap
          INTEGER(ip_) :: i, inf, it, itmax, k, kacc22, kbot, kdu, ks, kt,  &
            ktop, ku, kv, kwh, kwtop, kwv, ld, ls, lwkopt, ndec, ndfl, nh,  &
            nho, nibble, nmin, ns, nsmax, nsr, nve, nw, nwmax, nwr, nwupbd
          LOGICAL :: sorted
          CHARACTER :: jbcmpz*2
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: ILAENV2
          REAL(rp_) :: zdum(1, 1_ip_)
          EXTERNAL :: DLACPY, DLAHQR, DLANV2, DLAQR3, &
            DLAQR4, DLAQR5
          INTRINSIC :: ABS, REAL, INT, MAX, MIN, MOD
          info = 0
          IF (n==0) THEN
            work(1) = one
            RETURN
          END IF
          IF (n<=ntiny) THEN
            lwkopt = 1
            IF (lwork/=-1) CALL DLAHQR(wantt, wantz, n, ilo, ihi, h, ldh,   &
              wr, wi, iloz, ihiz, z, ldz, info)
          ELSE
            info = 0
            IF (wantt) THEN
              jbcmpz(1:1) = 'S'
            ELSE
              jbcmpz(1:1) = 'E'
            END IF
            IF (wantz) THEN
              jbcmpz(2:2) = 'V'
            ELSE
              jbcmpz(2:2) = 'N'
            END IF
            nwr = ILAENV2(13_ip_, 'LAQR0', jbcmpz, n, ilo, ihi, lwork)
            nwr = MAX( 2_ip_, nwr)
            nwr = MIN(ihi-ilo+1, (n-1)/3, nwr)
            nsr = ILAENV2(15_ip_, 'LAQR0', jbcmpz, n, ilo, ihi, lwork)
            nsr = MIN(nsr, (n-3)/6, ihi-ilo)
            nsr = MAX( 2_ip_, nsr-MOD(nsr,2))
            CALL DLAQR3(wantt, wantz, n, ilo, ihi, nwr+1, h, ldh, iloz,     &
              ihiz, z, ldz, ls, ld, wr, wi, h, ldh, n, h, ldh, n, h, ldh,   &
              work, -1_ip_)
            lwkopt = MAX(3*nsr/2, INT(work(1)))
            IF (lwork==-1) THEN
              work(1) = REAL(lwkopt,rp_)
              RETURN
            END IF
            nmin = ILAENV2(12_ip_, 'LAQR0', jbcmpz, n, ilo, ihi, lwork)
            nmin = MAX(ntiny, nmin)
            nibble = ILAENV2(14_ip_, 'LAQR0', jbcmpz, n, ilo, ihi, lwork)
            nibble = MAX(0, nibble)
            kacc22 = ILAENV2(16_ip_, 'LAQR0', jbcmpz, n, ilo, ihi, lwork)
            kacc22 = MAX(0, kacc22)
            kacc22 = MIN( 2_ip_, kacc22)
            nwmax = MIN((n-1)/3, lwork/2)
            nw = nwmax
            nsmax = MIN((n-3)/6, 2*lwork/3)
            nsmax = nsmax - MOD(nsmax, 2_ip_)
            ndfl = 1
            itmax = MAX(30, 2*kexsh)*MAX(10, (ihi-ilo+1))
            kbot = ihi
            DO it = 1, itmax
              IF (kbot<ilo) GO TO 90
              DO k = kbot, ilo + 1, -1_ip_
                IF (h(k,k-1)==zero) GO TO 20
              END DO
              k = ilo
 20           CONTINUE
              ktop = k
              nh = kbot - ktop + 1
              nwupbd = MIN(nh, nwmax)
              IF (ndfl<kexnw) THEN
                nw = MIN(nwupbd, nwr)
              ELSE
                nw = MIN(nwupbd, 2*nw)
              END IF
              IF (nw<nwmax) THEN
                IF (nw>=nh-1) THEN
                  nw = nh
                ELSE
                  kwtop = kbot - nw + 1
                  IF (ABS(h(kwtop,kwtop-1))>ABS(h(kwtop-1, kwtop-2))) nw =  &
                    nw + 1
                END IF
              END IF
              IF (ndfl<kexnw) THEN
                ndec = -1
              ELSE IF (ndec>=0 .OR. nw>=nwupbd) THEN
                ndec = ndec + 1
                IF (nw-ndec<2) ndec = 0
                nw = nw - ndec
              END IF
              kv = n - nw + 1
              kt = nw + 1
              nho = (n-nw-1) - kt + 1
              kwv = nw + 2
              nve = (n-nw) - kwv + 1
              CALL DLAQR3(wantt, wantz, n, ktop, kbot, nw, h, ldh, iloz,    &
                ihiz, z, ldz, ls, ld, wr, wi, h(kv,1), ldh, nho, h(kv,kt),  &
                ldh, nve, h(kwv,1), ldh, work, lwork)
              kbot = kbot - ld
              ks = kbot - ls + 1
              IF ((ld==0) .OR. ((100*ld<=nw*nibble) .AND. (kbot-ktop+1>     &
                MIN(nmin,nwmax)))) THEN
                ns = MIN(nsmax, nsr, MAX(2,kbot-ktop))
                ns = ns - MOD(ns, 2_ip_)
                IF (MOD(ndfl,kexsh)==0) THEN
                  ks = kbot - ns + 1
                  DO i = kbot, MAX(ks+1, ktop+2), -2
                    ss = ABS(h(i,i-1)) + ABS(h(i-1,i-2))
                    aa = wilk1*ss + h(i, i)
                    bb = ss
                    cc = wilk2*ss
                    dd = aa
                    CALL DLANV2(aa, bb, cc, dd, wr(i-1), wi(i-1), wr(i),    &
                      wi(i), cs, sn)
                  END DO
                  IF (ks==ktop) THEN
                    wr(ks+1) = h(ks+1, ks+1)
                    wi(ks+1) = zero
                    wr(ks) = wr(ks+1)
                    wi(ks) = wi(ks+1)
                  END IF
                ELSE
                  IF (kbot-ks+1<=ns/2) THEN
                    ks = kbot - ns + 1
                    kt = n - ns + 1
                    CALL DLACPY('A', ns, ns, h(ks,ks), ldh, h(kt,1), ldh)
                    IF (ns>nmin) THEN
                      CALL DLAQR4(.FALSE., .FALSE., ns, 1_ip_, ns, h(kt,1), &
                        ldh, wr(ks), wi(ks), 1_ip_, 1_ip_, zdum, 1_ip_, work,&
                        lwork, inf)
                    ELSE
                      CALL DLAHQR(.FALSE., .FALSE., ns, 1_ip_, ns, h(kt,1), &
                        ldh, wr(ks), wi(ks), 1_ip_, 1_ip_, zdum, 1_ip_, inf)
                    END IF
                    ks = ks + inf
                    IF (ks>=kbot) THEN
                      aa = h(kbot-1, kbot-1)
                      cc = h(kbot, kbot-1)
                      bb = h(kbot-1, kbot)
                      dd = h(kbot, kbot)
                      CALL DLANV2(aa, bb, cc, dd, wr(kbot-1), wi(kbot-1),   &
                        wr(kbot), wi(kbot), cs, sn)
                      ks = kbot - 1
                    END IF
                  END IF
                  IF (kbot-ks+1>ns) THEN
                    sorted = .FALSE.
                    DO k = kbot, ks + 1, -1_ip_
                      IF (sorted) GO TO 60
                      sorted = .TRUE.
                      DO i = ks, k - 1
                        IF (ABS(wr(i))+ABS(wi(i))<ABS(wr(i+1))+ABS(wi(i+    &
                          1))) THEN
                          sorted = .FALSE.
                          swap = wr(i)
                          wr(i) = wr(i+1)
                          wr(i+1) = swap
                          swap = wi(i)
                          wi(i) = wi(i+1)
                          wi(i+1) = swap
                        END IF
                      END DO
                    END DO
 60                 CONTINUE
                  END IF
                  DO i = kbot, ks + 2, -2
                    IF (wi(i)/=-wi(i-1)) THEN
                      swap = wr(i)
                      wr(i) = wr(i-1)
                      wr(i-1) = wr(i-2)
                      wr(i-2) = swap
                      swap = wi(i)
                      wi(i) = wi(i-1)
                      wi(i-1) = wi(i-2)
                      wi(i-2) = swap
                    END IF
                  END DO
                END IF
                IF (kbot-ks+1==2) THEN
                  IF (wi(kbot)==zero) THEN
                    IF (ABS(wr(kbot)-h(kbot,kbot))<ABS(wr(kbot-1)-h(kbot,   &
                      kbot))) THEN
                      wr(kbot-1) = wr(kbot)
                    ELSE
                      wr(kbot) = wr(kbot-1)
                    END IF
                  END IF
                END IF
                ns = MIN(ns, kbot-ks+1)
                ns = ns - MOD(ns, 2_ip_)
                ks = kbot - ns + 1
                kdu = 2*ns
                ku = n - kdu + 1
                kwh = kdu + 1
                nho = (n-kdu+1-4) - (kdu+1) + 1
                kwv = kdu + 4
                nve = n - kdu - kwv + 1
                CALL DLAQR5(wantt, wantz, kacc22, n, ktop, kbot, ns,        &
                  wr(ks), wi(ks), h, ldh, iloz, ihiz, z, ldz, work, 3_ip_,  &
                  h(ku,1), ldh, nve, h(kwv,1), ldh, nho, h(ku,kwh), ldh)
              END IF
              IF (ld>0) THEN
                ndfl = 1
              ELSE
                ndfl = ndfl + 1
              END IF
            END DO
            info = kbot
 90         CONTINUE
          END IF
          work(1) = REAL(lwkopt,rp_)
        END SUBROUTINE

        SUBROUTINE DLAQR1(n, h, ldh, sr1, si1, sr2, si2, v)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: si1, si2, sr1, sr2
          INTEGER(ip_) :: ldh, n
          REAL(rp_) :: h(ldh, *), v(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          REAL(rp_) :: h21s, h31s, s
          INTRINSIC :: ABS
          IF (n/=2 .AND. n/=3) THEN
            RETURN
          END IF
          IF (n==2) THEN
            s = ABS(h(1,1)-sr2) + ABS(si2) + ABS(h(2,1))
            IF (s==zero) THEN
              v(1) = zero
              v(2) = zero
            ELSE
              h21s = h( 2_ip_, 1_ip_)/s
              v(1) = h21s*h(1, 2_ip_) + (h(1,1)-sr1)*((h(1, 1_ip_)-sr2)/s)  &
                - si1*(si2/s)
              v(2) = h21s*(h(1,1)+h(2,2)-sr1-sr2)
            END IF
          ELSE
            s = ABS(h(1,1)-sr2) + ABS(si2) + ABS(h(2,1)) + ABS(h(3,1))
            IF (s==zero) THEN
              v(1) = zero
              v(2) = zero
              v(3) = zero
            ELSE
              h21s = h( 2_ip_, 1_ip_)/s
              h31s = h( 3_ip_, 1_ip_)/s
              v(1) = (h(1,1)-sr1)*((h(1,1)-sr2)/s) - si1*(si2/s) + h(1,     &
                2_ip_) *h21s + h(1, 3_ip_)*h31s
              v(2) = h21s*(h(1,1)+h(2,2)-sr1-sr2) + h( 2_ip_, 3_ip_)*h31s
              v(3) = h31s*(h(1,1)+h(3,3)-sr1-sr2) + h21s*h( 3_ip_, 2_ip_)
            END IF
          END IF
        END SUBROUTINE

        SUBROUTINE DLAQR2(wantt, wantz, n, ktop, kbot, nw, h, ldh, iloz,    &
          ihiz, z, ldz, ns, nd, sr, si, v, ldv, nh, t, ldt, nv, wv, ldwv,   &
          work, lwork)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: ihiz, iloz, kbot, ktop, ldh, ldt, ldv, ldwv, ldz, &
            lwork, n, nd, nh, ns, nv, nw
          LOGICAL :: wantt, wantz
          REAL(rp_) :: h(ldh, *), si(*), sr(*), t(ldt, *), v(ldv, *),       &
            work(*), wv(ldwv, *), z(ldz, *)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          REAL(rp_) :: aa, bb, beta, cc, cs, dd, evi, evk, foo, s, safmax,  &
            safmin, smlnum, sn, tau, ulp
          INTEGER(ip_) :: i, ifst, ilst, info, infqr, j, jw, k, kcol, kend, &
            kln, krow, kwtop, ltop, lwk1, lwk2, lwkopt
          LOGICAL :: bulge, sorted
          REAL(rp_) :: DLAMCH
          EXTERNAL :: DLAMCH
          EXTERNAL :: DCOPY, DGEHRD, DGEMM, DLABAD, DLACPY,& 
            DLAHQR, DLANV2, DLARF, DLARFG, DLASET, DORMHR, &
            DTREXC
          INTRINSIC :: ABS, REAL, INT, MAX, MIN, SQRT
          jw = MIN(nw, kbot-ktop+1)
          IF (jw<=2) THEN
            lwkopt = 1
          ELSE
            CALL DGEHRD(jw, 1_ip_, jw-1, t, ldt, work, work, -1_ip_, info)
            lwk1 = INT(work(1))
            CALL DORMHR('R', 'N', jw, jw, 1_ip_, jw-1, t, ldt, work, v,     &
              ldv, work, -1_ip_, info)
            lwk2 = INT(work(1))
            lwkopt = jw + MAX(lwk1, lwk2)
          END IF
          IF (lwork==-1) THEN
            work(1) = REAL(lwkopt,rp_)
            RETURN
          END IF
          ns = 0
          nd = 0
          work(1) = one
          IF (ktop>kbot) RETURN
          IF (nw<1) RETURN
          safmin = DLAMCH('SAFE MINIMUM')
          safmax = one/safmin
          CALL DLABAD(safmin, safmax)
          ulp = DLAMCH('PRECISION')
          smlnum = safmin*(REAL(n,rp_)/ulp)
          jw = MIN(nw, kbot-ktop+1)
          kwtop = kbot - jw + 1
          IF (kwtop==ktop) THEN
            s = zero
          ELSE
            s = h(kwtop, kwtop-1)
          END IF
          IF (kbot==kwtop) THEN
            sr(kwtop) = h(kwtop, kwtop)
            si(kwtop) = zero
            ns = 1
            nd = 0
            IF (ABS(s)<=MAX(smlnum,ulp*ABS(h(kwtop,kwtop)))) THEN
              ns = 0
              nd = 1
              IF (kwtop>ktop) h(kwtop, kwtop-1) = zero
            END IF
            work(1) = one
            RETURN
          END IF
          CALL DLACPY('U', jw, jw, h(kwtop,kwtop), ldh, t, ldt)
          CALL DCOPY(jw-1, h(kwtop+1,kwtop), ldh+1, t(2,1), ldt+1)
          CALL DLASET('A', jw, jw, zero, one, v, ldv)
          CALL DLAHQR(.TRUE., .TRUE., jw, 1_ip_, jw, t, ldt, sr(kwtop),     &
            si(kwtop), 1_ip_, jw, v, ldv, infqr)
          DO j = 1, jw - 3
            t(j+2, j) = zero
            t(j+3, j) = zero
          END DO
          IF (jw>2) t(jw, jw-2) = zero
          ns = jw
          ilst = infqr + 1
 20       CONTINUE
          IF (ilst<=ns) THEN
            IF (ns==1) THEN
              bulge = .FALSE.
            ELSE
              bulge = t(ns, ns-1) /= zero
            END IF
            IF (.NOT. bulge) THEN
              foo = ABS(t(ns,ns))
              IF (foo==zero) foo = ABS(s)
              IF (ABS(s*v(1,ns))<=MAX(smlnum,ulp*foo)) THEN
                ns = ns - 1
              ELSE
                ifst = ns
                CALL DTREXC('V', jw, t, ldt, v, ldv, ifst, ilst, work,      &
                  info)
                ilst = ilst + 1
              END IF
            ELSE
              foo = ABS(t(ns,ns)) + SQRT(ABS(t(ns,ns-1)))*SQRT(ABS(t(ns-1,  &
                ns)))
              IF (foo==zero) foo = ABS(s)
              IF (MAX(ABS(s*v(1,ns)),ABS(s*v(1,ns-1)))<=MAX(smlnum,ulp*foo  &
                )) THEN
                ns = ns - 2
              ELSE
                ifst = ns
                CALL DTREXC('V', jw, t, ldt, v, ldv, ifst, ilst, work,      &
                  info)
                ilst = ilst + 2
              END IF
            END IF
            GO TO 20
          END IF
          IF (ns==0) s = zero
          IF (ns<jw) THEN
            sorted = .FALSE.
            i = ns + 1
 30         CONTINUE
            IF (sorted) GO TO 50
            sorted = .TRUE.
            kend = i - 1
            i = infqr + 1
            IF (i==ns) THEN
              k = i + 1
            ELSE IF (t(i+1,i)==zero) THEN
              k = i + 1
            ELSE
              k = i + 2
            END IF
 40         CONTINUE
            IF (k<=kend) THEN
              IF (k==i+1) THEN
                evi = ABS(t(i,i))
              ELSE
                evi = ABS(t(i,i)) + SQRT(ABS(t(i+1,i)))*SQRT(ABS(t(i,       &
                  i+1)))
              END IF
              IF (k==kend) THEN
                evk = ABS(t(k,k))
              ELSE IF (t(k+1,k)==zero) THEN
                evk = ABS(t(k,k))
              ELSE
                evk = ABS(t(k,k)) + SQRT(ABS(t(k+1,k)))*SQRT(ABS(t(k,       &
                  k+1)))
              END IF
              IF (evi>=evk) THEN
                i = k
              ELSE
                sorted = .FALSE.
                ifst = i
                ilst = k
                CALL DTREXC('V', jw, t, ldt, v, ldv, ifst, ilst, work,      &
                  info)
                IF (info==0) THEN
                  i = ilst
                ELSE
                  i = k
                END IF
              END IF
              IF (i==kend) THEN
                k = i + 1
              ELSE IF (t(i+1,i)==zero) THEN
                k = i + 1
              ELSE
                k = i + 2
              END IF
              GO TO 40
            END IF
            GO TO 30
 50         CONTINUE
          END IF
          i = jw
 60       CONTINUE
          IF (i>=infqr+1) THEN
            IF (i==infqr+1) THEN
              sr(kwtop+i-1) = t(i, i)
              si(kwtop+i-1) = zero
              i = i - 1
            ELSE IF (t(i,i-1)==zero) THEN
              sr(kwtop+i-1) = t(i, i)
              si(kwtop+i-1) = zero
              i = i - 1
            ELSE
              aa = t(i-1, i-1)
              cc = t(i, i-1)
              bb = t(i-1, i)
              dd = t(i, i)
              CALL DLANV2(aa, bb, cc, dd, sr(kwtop+i-2), si(kwtop+i-2),     &
                sr(kwtop+i-1), si(kwtop+i-1), cs, sn)
              i = i - 2
            END IF
            GO TO 60
          END IF
          IF (ns<jw .OR. s==zero) THEN
            IF (ns>1 .AND. s/=zero) THEN
              CALL DCOPY(ns, v, ldv, work, 1_ip_)
              beta = work(1)
              CALL DLARFG(ns, beta, work(2), 1_ip_, tau)
              work(1) = one
              CALL DLASET('L', jw-2, jw-2, zero, zero, t(3,1), ldt)
              CALL DLARF('L', ns, jw, work, 1_ip_, tau, t, ldt, work(jw+1))
              CALL DLARF('R', ns, ns, work, 1_ip_, tau, t, ldt, work(jw+1))
              CALL DLARF('R', jw, ns, work, 1_ip_, tau, v, ldv, work(jw+1))
              CALL DGEHRD(jw, 1_ip_, ns, t, ldt, work, work(jw+1),          &
                lwork-jw, info)
            END IF
            IF (kwtop>1) h(kwtop, kwtop-1) = s*v(1, 1_ip_)
            CALL DLACPY('U', jw, jw, t, ldt, h(kwtop,kwtop), ldh)
            CALL DCOPY(jw-1, t(2,1), ldt+1, h(kwtop+1,kwtop), ldh+1)
            IF (ns>1 .AND. s/=zero) CALL DORMHR('R', 'N', jw, ns, 1_ip_,    &
              ns, t, ldt, work, v, ldv, work(jw+1), lwork-jw, info)
            IF (wantt) THEN
              ltop = 1
            ELSE
              ltop = ktop
            END IF
            DO krow = ltop, kwtop - 1, nv
              kln = MIN(nv, kwtop-krow)
              CALL DGEMM('N', 'N', kln, jw, jw, one, h(krow,kwtop), ldh, v, &
                ldv, zero, wv, ldwv)
              CALL DLACPY('A', kln, jw, wv, ldwv, h(krow,kwtop), ldh)
            END DO
            IF (wantt) THEN
              DO kcol = kbot + 1, n, nh
                kln = MIN(nh, n-kcol+1)
                CALL DGEMM('C', 'N', jw, kln, jw, one, v, ldv, h(kwtop,     &
                  kcol), ldh, zero, t, ldt)
                CALL DLACPY('A', jw, kln, t, ldt, h(kwtop,kcol), ldh)
              END DO
            END IF
            IF (wantz) THEN
              DO krow = iloz, ihiz, nv
                kln = MIN(nv, ihiz-krow+1)
                CALL DGEMM('N', 'N', kln, jw, jw, one, z(krow,kwtop), ldz,  &
                  v, ldv, zero, wv, ldwv)
                CALL DLACPY('A', kln, jw, wv, ldwv, z(krow,kwtop), ldz)
              END DO
            END IF
          END IF
          nd = jw - ns
          ns = ns - infqr
          work(1) = REAL(lwkopt,rp_)
        END SUBROUTINE

        SUBROUTINE DLAQR3(wantt, wantz, n, ktop, kbot, nw, h, ldh, iloz,    &
          ihiz, z, ldz, ns, nd, sr, si, v, ldv, nh, t, ldt, nv, wv, ldwv,   &
          work, lwork)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: ihiz, iloz, kbot, ktop, ldh, ldt, ldv, ldwv, ldz, &
            lwork, n, nd, nh, ns, nv, nw
          LOGICAL :: wantt, wantz
          REAL(rp_) :: h(ldh, *), si(*), sr(*), t(ldt, *), v(ldv, *),       &
            work(*), wv(ldwv, *), z(ldz, *)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          REAL(rp_) :: aa, bb, beta, cc, cs, dd, evi, evk, foo, s, safmax,  &
            safmin, smlnum, sn, tau, ulp
          INTEGER(ip_) :: i, ifst, ilst, info, infqr, j, jw, k, kcol, kend, &
            kln, krow, kwtop, ltop, lwk1, lwk2, lwk3, lwkopt, nmin
          LOGICAL :: bulge, sorted
          REAL(rp_) :: DLAMCH
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: DLAMCH, ILAENV2
          EXTERNAL :: DCOPY, DGEHRD, DGEMM, DLABAD, DLACPY,& 
            DLAHQR, DLANV2, DLAQR4, DLARF, DLARFG, DLASET, &
            DORMHR, DTREXC
          INTRINSIC :: ABS, REAL, INT, MAX, MIN, SQRT
          jw = MIN(nw, kbot-ktop+1)
          IF (jw<=2) THEN
            lwkopt = 1
          ELSE
            CALL DGEHRD(jw, 1_ip_, jw-1, t, ldt, work, work, -1_ip_, info)
            lwk1 = INT(work(1))
            CALL DORMHR('R', 'N', jw, jw, 1_ip_, jw-1, t, ldt, work, v,     &
              ldv, work, -1_ip_, info)
            lwk2 = INT(work(1))
            CALL DLAQR4(.TRUE., .TRUE., jw, 1_ip_, jw, t, ldt, sr, si,      &
              1_ip_, jw, v, ldv, work, -1_ip_, infqr)
            lwk3 = INT(work(1))
            lwkopt = MAX(jw+MAX(lwk1,lwk2), lwk3)
          END IF
          IF (lwork==-1) THEN
            work(1) = REAL(lwkopt,rp_)
            RETURN
          END IF
          ns = 0
          nd = 0
          work(1) = one
          IF (ktop>kbot) RETURN
          IF (nw<1) RETURN
          safmin = DLAMCH('SAFE MINIMUM')
          safmax = one/safmin
          CALL DLABAD(safmin, safmax)
          ulp = DLAMCH('PRECISION')
          smlnum = safmin*(REAL(n,rp_)/ulp)
          jw = MIN(nw, kbot-ktop+1)
          kwtop = kbot - jw + 1
          IF (kwtop==ktop) THEN
            s = zero
          ELSE
            s = h(kwtop, kwtop-1)
          END IF
          IF (kbot==kwtop) THEN
            sr(kwtop) = h(kwtop, kwtop)
            si(kwtop) = zero
            ns = 1
            nd = 0
            IF (ABS(s)<=MAX(smlnum,ulp*ABS(h(kwtop,kwtop)))) THEN
              ns = 0
              nd = 1
              IF (kwtop>ktop) h(kwtop, kwtop-1) = zero
            END IF
            work(1) = one
            RETURN
          END IF
          CALL DLACPY('U', jw, jw, h(kwtop,kwtop), ldh, t, ldt)
          CALL DCOPY(jw-1, h(kwtop+1,kwtop), ldh+1, t(2,1), ldt+1)
          CALL DLASET('A', jw, jw, zero, one, v, ldv)
          nmin = ILAENV2(12_ip_, 'LAQR3', 'SV', jw, 1_ip_, jw, lwork)
          IF (jw>nmin) THEN
            CALL DLAQR4(.TRUE., .TRUE., jw, 1_ip_, jw, t, ldt, sr(kwtop),   &
              si(kwtop), 1_ip_, jw, v, ldv, work, lwork, infqr)
          ELSE
            CALL DLAHQR(.TRUE., .TRUE., jw, 1_ip_, jw, t, ldt, sr(kwtop),   &
              si(kwtop), 1_ip_, jw, v, ldv, infqr)
          END IF
          DO j = 1, jw - 3
            t(j+2, j) = zero
            t(j+3, j) = zero
          END DO
          IF (jw>2) t(jw, jw-2) = zero
          ns = jw
          ilst = infqr + 1
 20       CONTINUE
          IF (ilst<=ns) THEN
            IF (ns==1) THEN
              bulge = .FALSE.
            ELSE
              bulge = t(ns, ns-1) /= zero
            END IF
            IF (.NOT. bulge) THEN
              foo = ABS(t(ns,ns))
              IF (foo==zero) foo = ABS(s)
              IF (ABS(s*v(1,ns))<=MAX(smlnum,ulp*foo)) THEN
                ns = ns - 1
              ELSE
                ifst = ns
                CALL DTREXC('V', jw, t, ldt, v, ldv, ifst, ilst, work,      &
                  info)
                ilst = ilst + 1
              END IF
            ELSE
              foo = ABS(t(ns,ns)) + SQRT(ABS(t(ns,ns-1)))*SQRT(ABS(t(ns-1,  &
                ns)))
              IF (foo==zero) foo = ABS(s)
              IF (MAX(ABS(s*v(1,ns)),ABS(s*v(1,ns-1)))<=MAX(smlnum,ulp*foo  &
                )) THEN
                ns = ns - 2
              ELSE
                ifst = ns
                CALL DTREXC('V', jw, t, ldt, v, ldv, ifst, ilst, work,      &
                  info)
                ilst = ilst + 2
              END IF
            END IF
            GO TO 20
          END IF
          IF (ns==0) s = zero
          IF (ns<jw) THEN
            sorted = .FALSE.
            i = ns + 1
 30         CONTINUE
            IF (sorted) GO TO 50
            sorted = .TRUE.
            kend = i - 1
            i = infqr + 1
            IF (i==ns) THEN
              k = i + 1
            ELSE IF (t(i+1,i)==zero) THEN
              k = i + 1
            ELSE
              k = i + 2
            END IF
 40         CONTINUE
            IF (k<=kend) THEN
              IF (k==i+1) THEN
                evi = ABS(t(i,i))
              ELSE
                evi = ABS(t(i,i)) + SQRT(ABS(t(i+1,i)))*SQRT(ABS(t(i,       &
                  i+1)))
              END IF
              IF (k==kend) THEN
                evk = ABS(t(k,k))
              ELSE IF (t(k+1,k)==zero) THEN
                evk = ABS(t(k,k))
              ELSE
                evk = ABS(t(k,k)) + SQRT(ABS(t(k+1,k)))*SQRT(ABS(t(k,       &
                  k+1)))
              END IF
              IF (evi>=evk) THEN
                i = k
              ELSE
                sorted = .FALSE.
                ifst = i
                ilst = k
                CALL DTREXC('V', jw, t, ldt, v, ldv, ifst, ilst, work,      &
                  info)
                IF (info==0) THEN
                  i = ilst
                ELSE
                  i = k
                END IF
              END IF
              IF (i==kend) THEN
                k = i + 1
              ELSE IF (t(i+1,i)==zero) THEN
                k = i + 1
              ELSE
                k = i + 2
              END IF
              GO TO 40
            END IF
            GO TO 30
 50         CONTINUE
          END IF
          i = jw
 60       CONTINUE
          IF (i>=infqr+1) THEN
            IF (i==infqr+1) THEN
              sr(kwtop+i-1) = t(i, i)
              si(kwtop+i-1) = zero
              i = i - 1
            ELSE IF (t(i,i-1)==zero) THEN
              sr(kwtop+i-1) = t(i, i)
              si(kwtop+i-1) = zero
              i = i - 1
            ELSE
              aa = t(i-1, i-1)
              cc = t(i, i-1)
              bb = t(i-1, i)
              dd = t(i, i)
              CALL DLANV2(aa, bb, cc, dd, sr(kwtop+i-2), si(kwtop+i-2),     &
                sr(kwtop+i-1), si(kwtop+i-1), cs, sn)
              i = i - 2
            END IF
            GO TO 60
          END IF
          IF (ns<jw .OR. s==zero) THEN
            IF (ns>1 .AND. s/=zero) THEN
              CALL DCOPY(ns, v, ldv, work, 1_ip_)
              beta = work(1)
              CALL DLARFG(ns, beta, work(2), 1_ip_, tau)
              work(1) = one
              CALL DLASET('L', jw-2, jw-2, zero, zero, t(3,1), ldt)
              CALL DLARF('L', ns, jw, work, 1_ip_, tau, t, ldt, work(jw+1))
              CALL DLARF('R', ns, ns, work, 1_ip_, tau, t, ldt, work(jw+1))
              CALL DLARF('R', jw, ns, work, 1_ip_, tau, v, ldv, work(jw+1))
              CALL DGEHRD(jw, 1_ip_, ns, t, ldt, work, work(jw+1),          &
                lwork-jw, info)
            END IF
            IF (kwtop>1) h(kwtop, kwtop-1) = s*v(1, 1_ip_)
            CALL DLACPY('U', jw, jw, t, ldt, h(kwtop,kwtop), ldh)
            CALL DCOPY(jw-1, t(2,1), ldt+1, h(kwtop+1,kwtop), ldh+1)
            IF (ns>1 .AND. s/=zero) CALL DORMHR('R', 'N', jw, ns, 1_ip_,    &
              ns, t, ldt, work, v, ldv, work(jw+1), lwork-jw, info)
            IF (wantt) THEN
              ltop = 1
            ELSE
              ltop = ktop
            END IF
            DO krow = ltop, kwtop - 1, nv
              kln = MIN(nv, kwtop-krow)
              CALL DGEMM('N', 'N', kln, jw, jw, one, h(krow,kwtop), ldh, v, &
                ldv, zero, wv, ldwv)
              CALL DLACPY('A', kln, jw, wv, ldwv, h(krow,kwtop), ldh)
            END DO
            IF (wantt) THEN
              DO kcol = kbot + 1, n, nh
                kln = MIN(nh, n-kcol+1)
                CALL DGEMM('C', 'N', jw, kln, jw, one, v, ldv, h(kwtop,     &
                  kcol), ldh, zero, t, ldt)
                CALL DLACPY('A', jw, kln, t, ldt, h(kwtop,kcol), ldh)
              END DO
            END IF
            IF (wantz) THEN
              DO krow = iloz, ihiz, nv
                kln = MIN(nv, ihiz-krow+1)
                CALL DGEMM('N', 'N', kln, jw, jw, one, z(krow,kwtop), ldz,  &
                  v, ldv, zero, wv, ldwv)
                CALL DLACPY('A', kln, jw, wv, ldwv, z(krow,kwtop), ldz)
              END DO
            END IF
          END IF
          nd = jw - ns
          ns = ns - infqr
          work(1) = REAL(lwkopt,rp_)
        END SUBROUTINE

        SUBROUTINE DLAQR4(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, iloz,  &
          ihiz, z, ldz, work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: ihi, ihiz, ilo, iloz, info, ldh, ldz, lwork, n
          LOGICAL :: wantt, wantz
          REAL(rp_) :: h(ldh, *), wi(*), work(*), wr(*), z(ldz, *)
          INTEGER(ip_) :: ntiny
          PARAMETER (ntiny=15)
          INTEGER(ip_) :: kexnw
          PARAMETER (kexnw=5)
          INTEGER(ip_) :: kexsh
          PARAMETER (kexsh=6)
          REAL(rp_) :: wilk1, wilk2
          PARAMETER (wilk1=0.75_rp_, wilk2=-0.4375_rp_)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          REAL(rp_) :: aa, bb, cc, cs, dd, sn, ss, swap
          INTEGER(ip_) :: i, inf, it, itmax, k, kacc22, kbot, kdu, ks, kt,  &
            ktop, ku, kv, kwh, kwtop, kwv, ld, ls, lwkopt, ndec, ndfl, nh,  &
            nho, nibble, nmin, ns, nsmax, nsr, nve, nw, nwmax, nwr, nwupbd
          LOGICAL :: sorted
          CHARACTER :: jbcmpz*2
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: ILAENV2
          REAL(rp_) :: zdum(1, 1_ip_)
          EXTERNAL :: DLACPY, DLAHQR, DLANV2, DLAQR2, &
            DLAQR5
          INTRINSIC :: ABS, REAL, INT, MAX, MIN, MOD
          info = 0
          IF (n==0) THEN
            work(1) = one
            RETURN
          END IF
          IF (n<=ntiny) THEN
            lwkopt = 1
            IF (lwork/=-1) CALL DLAHQR(wantt, wantz, n, ilo, ihi, h, ldh,   &
              wr, wi, iloz, ihiz, z, ldz, info)
          ELSE
            info = 0
            IF (wantt) THEN
              jbcmpz(1:1) = 'S'
            ELSE
              jbcmpz(1:1) = 'E'
            END IF
            IF (wantz) THEN
              jbcmpz(2:2) = 'V'
            ELSE
              jbcmpz(2:2) = 'N'
            END IF
            nwr = ILAENV2(13_ip_, 'LAQR4', jbcmpz, n, ilo, ihi, lwork)
            nwr = MAX( 2_ip_, nwr)
            nwr = MIN(ihi-ilo+1, (n-1)/3, nwr)
            nsr = ILAENV2(15_ip_, 'LAQR4', jbcmpz, n, ilo, ihi, lwork)
            nsr = MIN(nsr, (n-3)/6, ihi-ilo)
            nsr = MAX( 2_ip_, nsr-MOD(nsr,2))
            CALL DLAQR2(wantt, wantz, n, ilo, ihi, nwr+1, h, ldh, iloz,     &
              ihiz, z, ldz, ls, ld, wr, wi, h, ldh, n, h, ldh, n, h, ldh,   &
              work, -1_ip_)
            lwkopt = MAX(3*nsr/2, INT(work(1)))
            IF (lwork==-1) THEN
              work(1) = REAL(lwkopt,rp_)
              RETURN
            END IF
            nmin = ILAENV2(12_ip_, 'LAQR4', jbcmpz, n, ilo, ihi, lwork)
            nmin = MAX(ntiny, nmin)
            nibble = ILAENV2(14_ip_, 'LAQR4', jbcmpz, n, ilo, ihi, lwork)
            nibble = MAX(0, nibble)
            kacc22 = ILAENV2(16_ip_, 'LAQR4', jbcmpz, n, ilo, ihi, lwork)
            kacc22 = MAX(0, kacc22)
            kacc22 = MIN( 2_ip_, kacc22)
            nwmax = MIN((n-1)/3, lwork/2)
            nw = nwmax
            nsmax = MIN((n-3)/6, 2*lwork/3)
            nsmax = nsmax - MOD(nsmax, 2_ip_)
            ndfl = 1
            itmax = MAX(30, 2*kexsh)*MAX(10, (ihi-ilo+1))
            kbot = ihi
            DO it = 1, itmax
              IF (kbot<ilo) GO TO 90
              DO k = kbot, ilo + 1, -1_ip_
                IF (h(k,k-1)==zero) GO TO 20
              END DO
              k = ilo
 20           CONTINUE
              ktop = k
              nh = kbot - ktop + 1
              nwupbd = MIN(nh, nwmax)
              IF (ndfl<kexnw) THEN
                nw = MIN(nwupbd, nwr)
              ELSE
                nw = MIN(nwupbd, 2*nw)
              END IF
              IF (nw<nwmax) THEN
                IF (nw>=nh-1) THEN
                  nw = nh
                ELSE
                  kwtop = kbot - nw + 1
                  IF (ABS(h(kwtop,kwtop-1))>ABS(h(kwtop-1, kwtop-2))) nw =  &
                    nw + 1
                END IF
              END IF
              IF (ndfl<kexnw) THEN
                ndec = -1
              ELSE IF (ndec>=0 .OR. nw>=nwupbd) THEN
                ndec = ndec + 1
                IF (nw-ndec<2) ndec = 0
                nw = nw - ndec
              END IF
              kv = n - nw + 1
              kt = nw + 1
              nho = (n-nw-1) - kt + 1
              kwv = nw + 2
              nve = (n-nw) - kwv + 1
              CALL DLAQR2(wantt, wantz, n, ktop, kbot, nw, h, ldh, iloz,    &
                ihiz, z, ldz, ls, ld, wr, wi, h(kv,1), ldh, nho, h(kv,kt),  &
                ldh, nve, h(kwv,1), ldh, work, lwork)
              kbot = kbot - ld
              ks = kbot - ls + 1
              IF ((ld==0) .OR. ((100*ld<=nw*nibble) .AND. (kbot-ktop+1>     &
                MIN(nmin,nwmax)))) THEN
                ns = MIN(nsmax, nsr, MAX(2,kbot-ktop))
                ns = ns - MOD(ns, 2_ip_)
                IF (MOD(ndfl,kexsh)==0) THEN
                  ks = kbot - ns + 1
                  DO i = kbot, MAX(ks+1, ktop+2), -2
                    ss = ABS(h(i,i-1)) + ABS(h(i-1,i-2))
                    aa = wilk1*ss + h(i, i)
                    bb = ss
                    cc = wilk2*ss
                    dd = aa
                    CALL DLANV2(aa, bb, cc, dd, wr(i-1), wi(i-1), wr(i),    &
                      wi(i), cs, sn)
                  END DO
                  IF (ks==ktop) THEN
                    wr(ks+1) = h(ks+1, ks+1)
                    wi(ks+1) = zero
                    wr(ks) = wr(ks+1)
                    wi(ks) = wi(ks+1)
                  END IF
                ELSE
                  IF (kbot-ks+1<=ns/2) THEN
                    ks = kbot - ns + 1
                    kt = n - ns + 1
                    CALL DLACPY('A', ns, ns, h(ks,ks), ldh, h(kt,1), ldh)
                    CALL DLAHQR(.FALSE., .FALSE., ns, 1_ip_, ns, h(kt,1),   &
                      ldh, wr(ks), wi(ks), 1_ip_, 1_ip_, zdum, 1_ip_, inf)
                    ks = ks + inf
                    IF (ks>=kbot) THEN
                      aa = h(kbot-1, kbot-1)
                      cc = h(kbot, kbot-1)
                      bb = h(kbot-1, kbot)
                      dd = h(kbot, kbot)
                      CALL DLANV2(aa, bb, cc, dd, wr(kbot-1), wi(kbot-1),   &
                        wr(kbot), wi(kbot), cs, sn)
                      ks = kbot - 1
                    END IF
                  END IF
                  IF (kbot-ks+1>ns) THEN
                    sorted = .FALSE.
                    DO k = kbot, ks + 1, -1_ip_
                      IF (sorted) GO TO 60
                      sorted = .TRUE.
                      DO i = ks, k - 1
                        IF (ABS(wr(i))+ABS(wi(i))<ABS(wr(i+1))+ABS(wi(i+    &
                          1))) THEN
                          sorted = .FALSE.
                          swap = wr(i)
                          wr(i) = wr(i+1)
                          wr(i+1) = swap
                          swap = wi(i)
                          wi(i) = wi(i+1)
                          wi(i+1) = swap
                        END IF
                      END DO
                    END DO
 60                 CONTINUE
                  END IF
                  DO i = kbot, ks + 2, -2
                    IF (wi(i)/=-wi(i-1)) THEN
                      swap = wr(i)
                      wr(i) = wr(i-1)
                      wr(i-1) = wr(i-2)
                      wr(i-2) = swap
                      swap = wi(i)
                      wi(i) = wi(i-1)
                      wi(i-1) = wi(i-2)
                      wi(i-2) = swap
                    END IF
                  END DO
                END IF
                IF (kbot-ks+1==2) THEN
                  IF (wi(kbot)==zero) THEN
                    IF (ABS(wr(kbot)-h(kbot,kbot))<ABS(wr(kbot-1)-h(kbot,   &
                      kbot))) THEN
                      wr(kbot-1) = wr(kbot)
                    ELSE
                      wr(kbot) = wr(kbot-1)
                    END IF
                  END IF
                END IF
                ns = MIN(ns, kbot-ks+1)
                ns = ns - MOD(ns, 2_ip_)
                ks = kbot - ns + 1
                kdu = 2*ns
                ku = n - kdu + 1
                kwh = kdu + 1
                nho = (n-kdu+1-4) - (kdu+1) + 1
                kwv = kdu + 4
                nve = n - kdu - kwv + 1
                CALL DLAQR5(wantt, wantz, kacc22, n, ktop, kbot, ns,        &
                  wr(ks), wi(ks), h, ldh, iloz, ihiz, z, ldz, work, 3_ip_,  &
                  h(ku,1), ldh, nve, h(kwv,1), ldh, nho, h(ku,kwh), ldh)
              END IF
              IF (ld>0) THEN
                ndfl = 1
              ELSE
                ndfl = ndfl + 1
              END IF
            END DO
            info = kbot
 90         CONTINUE
          END IF
          work(1) = REAL(lwkopt,rp_)
        END SUBROUTINE

        SUBROUTINE DLAQR5(wantt, wantz, kacc22, n, ktop, kbot, nshfts, sr,  &
          si, h, ldh, iloz, ihiz, z, ldz, v, ldv, u, ldu, nv, wv, ldwv, nh, &
          wh, ldwh)
          USE BLAS_LAPACK_KINDS_precision
          IMPLICIT NONE
          INTEGER(ip_) :: ihiz, iloz, kacc22, kbot, ktop, ldh, ldu, ldv,    &
            ldwh, ldwv, ldz, n, nh, nshfts, nv
          LOGICAL :: wantt, wantz
          REAL(rp_) :: h(ldh, *), si(*), sr(*), u(ldu, *), v(ldv, *),       &
            wh(ldwh, *), wv(ldwv, *), z(ldz, *)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          REAL(rp_) :: alpha, beta, h11, h12, h21, h22, refsum, safmax,     &
            safmin, scl, smlnum, swap, tst1, tst2, ulp
          INTEGER(ip_) :: i, i2, i4, incol, j, jbot, jcol, jlen, jrow,      &
            jtop, k, k1, kdu, kms, krcol, m, m22, mbot, mtop, nbmps, ndcol, &
            ns, nu
          LOGICAL :: accum, bmp22
          REAL(rp_) :: DLAMCH
          EXTERNAL :: DLAMCH
          INTRINSIC :: ABS, REAL, MAX, MIN, MOD
          REAL(rp_) :: vt(3)
          EXTERNAL :: DGEMM, DLABAD, DLACPY, DLAQR1, &
            DLARFG, DLASET, DTRMM
          IF (nshfts<2) RETURN
          IF (ktop>=kbot) RETURN
          DO i = 1, nshfts - 2, 2
            IF (si(i)/=-si(i+1)) THEN
              swap = sr(i)
              sr(i) = sr(i+1)
              sr(i+1) = sr(i+2)
              sr(i+2) = swap
              swap = si(i)
              si(i) = si(i+1)
              si(i+1) = si(i+2)
              si(i+2) = swap
            END IF
          END DO
          ns = nshfts - MOD(nshfts, 2_ip_)
          safmin = DLAMCH('SAFE MINIMUM')
          safmax = one/safmin
          CALL DLABAD(safmin, safmax)
          ulp = DLAMCH('PRECISION')
          smlnum = safmin*(REAL(n,rp_)/ulp)
          accum = (kacc22==1) .OR. (kacc22==2)
          IF (ktop+2<=kbot) h(ktop+2, ktop) = zero
          nbmps = ns/2
          kdu = 4*nbmps
          DO incol = ktop - 2*nbmps + 1, kbot - 2, 2*nbmps
            IF (accum) THEN
              jtop = MAX(ktop, incol)
            ELSE IF (wantt) THEN
              jtop = 1
            ELSE
              jtop = ktop
            END IF
            ndcol = incol + kdu
            IF (accum) CALL DLASET('ALL', kdu, kdu, zero, one, u, ldu)
            DO krcol = incol, MIN(incol+2*nbmps-1, kbot-2)
              mtop = MAX(1, (ktop-krcol)/2+1)
              mbot = MIN(nbmps, (kbot-krcol-1)/2)
              m22 = mbot + 1
              bmp22 = (mbot<nbmps) .AND. (krcol+2*(m22-1)) == (kbot-2)
              IF (bmp22) THEN
                k = krcol + 2*(m22-1)
                IF (k==ktop-1) THEN
                  CALL DLAQR1( 2_ip_, h(k+1,k+1), ldh, sr(2*m22-1),         &
                    si(2*m22-1), sr(2*m22), si(2*m22), v(1,m22))
                  beta = v(1, m22)
                  CALL DLARFG( 2_ip_, beta, v(2,m22), 1_ip_, v(1,m22))
                ELSE
                  beta = h(k+1, k)
                  v( 2_ip_, m22) = h(k+2, k)
                  CALL DLARFG( 2_ip_, beta, v(2,m22), 1_ip_, v(1,m22))
                  h(k+1, k) = beta
                  h(k+2, k) = zero
                END IF
                DO j = jtop, MIN(kbot, k+3)
                  refsum = v(1, m22)*(h(j,k+1)+v(2,m22)*h(j,k+2))
                  h(j, k+1) = h(j, k+1) - refsum
                  h(j, k+2) = h(j, k+2) - refsum*v( 2_ip_, m22)
                END DO
                IF (accum) THEN
                  jbot = MIN(ndcol, kbot)
                ELSE IF (wantt) THEN
                  jbot = n
                ELSE
                  jbot = kbot
                END IF
                DO j = k + 1, jbot
                  refsum = v(1, m22)*(h(k+1,j)+v(2,m22)*h(k+2,j))
                  h(k+1, j) = h(k+1, j) - refsum
                  h(k+2, j) = h(k+2, j) - refsum*v( 2_ip_, m22)
                END DO
                IF (k>=ktop) THEN
                  IF (h(k+1,k)/=zero) THEN
                    tst1 = ABS(h(k,k)) + ABS(h(k+1,k+1))
                    IF (tst1==zero) THEN
                      IF (k>=ktop+1) tst1 = tst1 + ABS(h(k,k-1))
                      IF (k>=ktop+2) tst1 = tst1 + ABS(h(k,k-2))
                      IF (k>=ktop+3) tst1 = tst1 + ABS(h(k,k-3))
                      IF (k<=kbot-2) tst1 = tst1 + ABS(h(k+2,k+1))
                      IF (k<=kbot-3) tst1 = tst1 + ABS(h(k+3,k+1))
                      IF (k<=kbot-4) tst1 = tst1 + ABS(h(k+4,k+1))
                    END IF
                    IF (ABS(h(k+1,k))<=MAX(smlnum,ulp*tst1)) THEN
                      h12 = MAX(ABS(h(k+1,k)), ABS(h(k,k+1)))
                      h21 = MIN(ABS(h(k+1,k)), ABS(h(k,k+1)))
                      h11 = MAX(ABS(h(k+1,k+1)), ABS(h(k,k)-h(k+1,k+1)))
                      h22 = MIN(ABS(h(k+1,k+1)), ABS(h(k,k)-h(k+1,k+1)))
                      scl = h11 + h12
                      tst2 = h22*(h11/scl)
                      IF (tst2==zero .OR. h21*(h12/scl)<=MAX(smlnum,ulp*    &
                        tst2)) THEN
                        h(k+1, k) = zero
                      END IF
                    END IF
                  END IF
                END IF
                IF (accum) THEN
                  kms = k - incol
                  DO j = MAX(1, ktop-incol), kdu
                    refsum = v(1, m22)*(u(j,kms+1)+v(2,m22)*u(j,kms+2))
                    u(j, kms+1) = u(j, kms+1) - refsum
                    u(j, kms+2) = u(j, kms+2) - refsum*v( 2_ip_, m22)
                  END DO
                ELSE IF (wantz) THEN
                  DO j = iloz, ihiz
                    refsum = v(1, m22)*(z(j,k+1)+v(2,m22)*z(j,k+2))
                    z(j, k+1) = z(j, k+1) - refsum
                    z(j, k+2) = z(j, k+2) - refsum*v( 2_ip_, m22)
                  END DO
                END IF
              END IF
              DO m = mbot, mtop, -1_ip_
                k = krcol + 2*(m-1)
                IF (k==ktop-1) THEN
                  CALL DLAQR1( 3_ip_, h(ktop,ktop), ldh, sr(2*m-1),         &
                    si(2*m-1), sr(2*m), si(2*m), v(1,m))
                  alpha = v(1, m)
                  CALL DLARFG( 3_ip_, alpha, v(2,m), 1_ip_, v(1,m))
                ELSE
                  refsum = v(1, m)*v( 3_ip_, m)*h(k+3, k+2)
                  h(k+3, k) = -refsum
                  h(k+3, k+1) = -refsum*v( 2_ip_, m)
                  h(k+3, k+2) = h(k+3, k+2) - refsum*v( 3_ip_, m)
                  beta = h(k+1, k)
                  v( 2_ip_, m) = h(k+2, k)
                  v( 3_ip_, m) = h(k+3, k)
                  CALL DLARFG( 3_ip_, beta, v(2,m), 1_ip_, v(1,m))
                  IF (h(k+3,k)/=zero .OR. h(k+3,k+1)/=zero .OR. h(k+3,      &
                    k+2)==zero) THEN
                    h(k+1, k) = beta
                    h(k+2, k) = zero
                    h(k+3, k) = zero
                  ELSE
                    CALL DLAQR1( 3_ip_, h(k+1,k+1), ldh, sr(2*m-1),         &
                      si(2*m-1), sr(2*m), si(2*m), vt)
                    alpha = vt(1)
                    CALL DLARFG( 3_ip_, alpha, vt(2), 1_ip_, vt(1))
                    refsum = vt(1)*(h(k+1,k)+vt(2)*h(k+2,k))
                    IF (ABS(h(k+2,k)-refsum*vt(2))+ABS(refsum*vt(3))>ulp*(  &
                      ABS(h(k,k))+ABS(h(k+1,k+1))+ABS(h(k+2,k+2)))) THEN
                      h(k+1, k) = beta
                      h(k+2, k) = zero
                      h(k+3, k) = zero
                    ELSE
                      h(k+1, k) = h(k+1, k) - refsum
                      h(k+2, k) = zero
                      h(k+3, k) = zero
                      v(1, m) = vt(1)
                      v( 2_ip_, m) = vt(2)
                      v( 3_ip_, m) = vt(3)
                    END IF
                  END IF
                END IF
                DO j = jtop, MIN(kbot, k+3)
                  refsum = v(1, m)*(h(j,k+1)+v(2,m)*h(j,k+2)+v(3,m)*h(j,k+  &
                    3))
                  h(j, k+1) = h(j, k+1) - refsum
                  h(j, k+2) = h(j, k+2) - refsum*v( 2_ip_, m)
                  h(j, k+3) = h(j, k+3) - refsum*v( 3_ip_, m)
                END DO
                refsum = v(1, m)*(h(k+1,k+1)+v(2,m)*h(k+2,k+1)+v(3,m)*h(k+  &
                  3,k+1))
                h(k+1, k+1) = h(k+1, k+1) - refsum
                h(k+2, k+1) = h(k+2, k+1) - refsum*v( 2_ip_, m)
                h(k+3, k+1) = h(k+3, k+1) - refsum*v( 3_ip_, m)
                IF (k<ktop) CYCLE
                IF (h(k+1,k)/=zero) THEN
                  tst1 = ABS(h(k,k)) + ABS(h(k+1,k+1))
                  IF (tst1==zero) THEN
                    IF (k>=ktop+1) tst1 = tst1 + ABS(h(k,k-1))
                    IF (k>=ktop+2) tst1 = tst1 + ABS(h(k,k-2))
                    IF (k>=ktop+3) tst1 = tst1 + ABS(h(k,k-3))
                    IF (k<=kbot-2) tst1 = tst1 + ABS(h(k+2,k+1))
                    IF (k<=kbot-3) tst1 = tst1 + ABS(h(k+3,k+1))
                    IF (k<=kbot-4) tst1 = tst1 + ABS(h(k+4,k+1))
                  END IF
                  IF (ABS(h(k+1,k))<=MAX(smlnum,ulp*tst1)) THEN
                    h12 = MAX(ABS(h(k+1,k)), ABS(h(k,k+1)))
                    h21 = MIN(ABS(h(k+1,k)), ABS(h(k,k+1)))
                    h11 = MAX(ABS(h(k+1,k+1)), ABS(h(k,k)-h(k+1,k+1)))
                    h22 = MIN(ABS(h(k+1,k+1)), ABS(h(k,k)-h(k+1,k+1)))
                    scl = h11 + h12
                    tst2 = h22*(h11/scl)
                    IF (tst2==zero .OR. h21*(h12/scl)<=MAX(smlnum,ulp*tst2  &
                      )) THEN
                      h(k+1, k) = zero
                    END IF
                  END IF
                END IF
              END DO
              IF (accum) THEN
                jbot = MIN(ndcol, kbot)
              ELSE IF (wantt) THEN
                jbot = n
              ELSE
                jbot = kbot
              END IF
              DO m = mbot, mtop, -1_ip_
                k = krcol + 2*(m-1)
                DO j = MAX(ktop, krcol+2*m), jbot
                  refsum = v(1, m)*(h(k+1,j)+v(2,m)*h(k+2,j)+v(3,m)*h(k+3,  &
                    j))
                  h(k+1, j) = h(k+1, j) - refsum
                  h(k+2, j) = h(k+2, j) - refsum*v( 2_ip_, m)
                  h(k+3, j) = h(k+3, j) - refsum*v( 3_ip_, m)
                END DO
              END DO
              IF (accum) THEN
                DO m = mbot, mtop, -1_ip_
                  k = krcol + 2*(m-1)
                  kms = k - incol
                  i2 = MAX(1, ktop-incol)
                  i2 = MAX(i2, kms-(krcol-incol)+1)
                  i4 = MIN(kdu, krcol+2*(mbot-1)-incol+5)
                  DO j = i2, i4
                    refsum = v(1, m)*(u(j,kms+1)+v(2,m)*u(j,kms+2)+v(3,m)*  &
                      u(j,kms+3))
                    u(j, kms+1) = u(j, kms+1) - refsum
                    u(j, kms+2) = u(j, kms+2) - refsum*v( 2_ip_, m)
                    u(j, kms+3) = u(j, kms+3) - refsum*v( 3_ip_, m)
                  END DO
                END DO
              ELSE IF (wantz) THEN
                DO m = mbot, mtop, -1_ip_
                  k = krcol + 2*(m-1)
                  DO j = iloz, ihiz
                    refsum = v(1, m)*(z(j,k+1)+v(2,m)*z(j,k+2)+v(3,m)*z(j,  &
                      k+3))
                    z(j, k+1) = z(j, k+1) - refsum
                    z(j, k+2) = z(j, k+2) - refsum*v( 2_ip_, m)
                    z(j, k+3) = z(j, k+3) - refsum*v( 3_ip_, m)
                  END DO
                END DO
              END IF
            END DO
            IF (accum) THEN
              IF (wantt) THEN
                jtop = 1
                jbot = n
              ELSE
                jtop = ktop
                jbot = kbot
              END IF
              k1 = MAX(1, ktop-incol)
              nu = (kdu-MAX(0,ndcol-kbot)) - k1 + 1
              DO jcol = MIN(ndcol, kbot) + 1, jbot, nh
                jlen = MIN(nh, jbot-jcol+1)
                CALL DGEMM('C', 'N', nu, jlen, nu, one, u(k1,k1), ldu,      &
                  h(incol+k1,jcol), ldh, zero, wh, ldwh)
                CALL DLACPY('ALL', nu, jlen, wh, ldwh, h(incol+k1,jcol),    &
                  ldh)
              END DO
              DO jrow = jtop, MAX(ktop, incol) - 1, nv
                jlen = MIN(nv, MAX(ktop,incol)-jrow)
                CALL DGEMM('N', 'N', jlen, nu, nu, one, h(jrow,incol+k1),   &
                  ldh, u(k1,k1), ldu, zero, wv, ldwv)
                CALL DLACPY('ALL', jlen, nu, wv, ldwv, h(jrow,incol+k1),    &
                  ldh)
              END DO
              IF (wantz) THEN
                DO jrow = iloz, ihiz, nv
                  jlen = MIN(nv, ihiz-jrow+1)
                  CALL DGEMM('N', 'N', jlen, nu, nu, one, z(jrow,incol+k1), &
                    ldz, u(k1,k1), ldu, zero, wv, ldwv)
                  CALL DLACPY('ALL', jlen, nu, wv, ldwv, z(jrow,incol+k1),  &
                    ldz)
                END DO
              END IF
            END IF
          END DO
        END SUBROUTINE

        SUBROUTINE DLARF(side, m, n, v, incv, tau, c, ldc, work)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: side
          INTEGER(ip_) :: incv, ldc, m, n
          REAL(rp_) :: tau
          REAL(rp_) :: c(ldc, *), v(*), work(*)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          LOGICAL :: applyleft
          INTEGER(ip_) :: i, lastv, lastc
          EXTERNAL :: DGEMV, DGER
          LOGICAL :: LSAME
          INTEGER(ip_) :: ILADLR, ILADLC
          EXTERNAL :: LSAME, ILADLR, ILADLC
          applyleft = LSAME(side, 'L')
          lastv = 0
          lastc = 0
          IF (tau/=zero) THEN
            IF (applyleft) THEN
              lastv = m
            ELSE
              lastv = n
            END IF
            IF (incv>0) THEN
              i = 1 + (lastv-1)*incv
            ELSE
              i = 1
            END IF
            DO WHILE (lastv>0 .AND. v(i)==zero)
              lastv = lastv - 1
              i = i - incv
            END DO
            IF (applyleft) THEN
              lastc = ILADLC(lastv, n, c, ldc)
            ELSE
              lastc = ILADLR(m, lastv, c, ldc)
            END IF
          END IF
          IF (applyleft) THEN
            IF (lastv>0) THEN
              CALL DGEMV('Transpose', lastv, lastc, one, c, ldc, v, incv,   &
                zero, work, 1_ip_)
              CALL DGER(lastv, lastc, -tau, v, incv, work, 1_ip_, c, ldc)
            END IF
          ELSE
            IF (lastv>0) THEN
              CALL DGEMV('No transpose', lastc, lastv, one, c, ldc, v,      &
                incv, zero, work, 1_ip_)
              CALL DGER(lastc, lastv, -tau, work, 1_ip_, v, incv, c, ldc)
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLARFB(side, trans, direct, storev, m, n, k, v, ldv, t,  &
          ldt, c, ldc, work, ldwork)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: direct, side, storev, trans
          INTEGER(ip_) :: k, ldc, ldt, ldv, ldwork, m, n
          REAL(rp_) :: c(ldc, *), t(ldt, *), v(ldv, *), work(ldwork, *)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          CHARACTER :: transt
          INTEGER(ip_) :: i, j
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: DCOPY, DGEMM, DTRMM
          IF (m<=0 .OR. n<=0) RETURN
          IF (LSAME(trans,'N')) THEN
            transt = 'T'
          ELSE
            transt = 'N'
          END IF
          IF (LSAME(storev,'C')) THEN
            IF (LSAME(direct,'F')) THEN
              IF (LSAME(side,'L')) THEN
                DO j = 1, k
                  CALL DCOPY(n, c(j,1), ldc, work(1,j), 1_ip_)
                END DO
                CALL DTRMM('Right', 'Lower', 'No transpose', 'Unit', n, k,  &
                  one, v, ldv, work, ldwork)
                IF (m>k) THEN
                  CALL DGEMM('Transpose', 'No transpose', n, k, m-k, one,   &
                    c(k+1,1), ldc, v(k+1,1), ldv, one, work, ldwork)
                END IF
                CALL DTRMM('Right', 'Upper', transt, 'Non-unit', n, k, one, &
                  t, ldt, work, ldwork)
                IF (m>k) THEN
                  CALL DGEMM('No transpose', 'Transpose', m-k, n, k, -one,  &
                    v(k+1,1), ldv, work, ldwork, one, c(k+1,1), ldc)
                END IF
                CALL DTRMM('Right', 'Lower', 'Transpose', 'Unit', n, k,     &
                  one, v, ldv, work, ldwork)
                DO j = 1, k
                  DO i = 1, n
                    c(j, i) = c(j, i) - work(i, j)
                  END DO
                END DO
              ELSE IF (LSAME(side,'R')) THEN
                DO j = 1, k
                  CALL DCOPY(m, c(1,j), 1_ip_, work(1,j), 1_ip_)
                END DO
                CALL DTRMM('Right', 'Lower', 'No transpose', 'Unit', m, k,  &
                  one, v, ldv, work, ldwork)
                IF (n>k) THEN
                  CALL DGEMM('No transpose', 'No transpose', m, k, n-k,     &
                    one, c(1,k+1), ldc, v(k+1,1), ldv, one, work, ldwork)
                END IF
                CALL DTRMM('Right', 'Upper', trans, 'Non-unit', m, k, one,  &
                  t, ldt, work, ldwork)
                IF (n>k) THEN
                  CALL DGEMM('No transpose', 'Transpose', m, n-k, k, -one,  &
                    work, ldwork, v(k+1,1), ldv, one, c(1,k+1), ldc)
                END IF
                CALL DTRMM('Right', 'Lower', 'Transpose', 'Unit', m, k,     &
                  one, v, ldv, work, ldwork)
                DO j = 1, k
                  DO i = 1, m
                    c(i, j) = c(i, j) - work(i, j)
                  END DO
                END DO
              END IF
            ELSE
              IF (LSAME(side,'L')) THEN
                DO j = 1, k
                  CALL DCOPY(n, c(m-k+j,1), ldc, work(1,j), 1_ip_)
                END DO
                CALL DTRMM('Right', 'Upper', 'No transpose', 'Unit', n, k,  &
                  one, v(m-k+1,1), ldv, work, ldwork)
                IF (m>k) THEN
                  CALL DGEMM('Transpose', 'No transpose', n, k, m-k, one,   &
                    c, ldc, v, ldv, one, work, ldwork)
                END IF
                CALL DTRMM('Right', 'Lower', transt, 'Non-unit', n, k, one, &
                  t, ldt, work, ldwork)
                IF (m>k) THEN
                  CALL DGEMM('No transpose', 'Transpose', m-k, n, k, -one,  &
                    v, ldv, work, ldwork, one, c, ldc)
                END IF
                CALL DTRMM('Right', 'Upper', 'Transpose', 'Unit', n, k,     &
                  one, v(m-k+1,1), ldv, work, ldwork)
                DO j = 1, k
                  DO i = 1, n
                    c(m-k+j, i) = c(m-k+j, i) - work(i, j)
                  END DO
                END DO
              ELSE IF (LSAME(side,'R')) THEN
                DO j = 1, k
                  CALL DCOPY(m, c(1,n-k+j), 1_ip_, work(1,j), 1_ip_)
                END DO
                CALL DTRMM('Right', 'Upper', 'No transpose', 'Unit', m, k,  &
                  one, v(n-k+1,1), ldv, work, ldwork)
                IF (n>k) THEN
                  CALL DGEMM('No transpose', 'No transpose', m, k, n-k,     &
                    one, c, ldc, v, ldv, one, work, ldwork)
                END IF
                CALL DTRMM('Right', 'Lower', trans, 'Non-unit', m, k, one,  &
                  t, ldt, work, ldwork)
                IF (n>k) THEN
                  CALL DGEMM('No transpose', 'Transpose', m, n-k, k, -one,  &
                    work, ldwork, v, ldv, one, c, ldc)
                END IF
                CALL DTRMM('Right', 'Upper', 'Transpose', 'Unit', m, k,     &
                  one, v(n-k+1,1), ldv, work, ldwork)
                DO j = 1, k
                  DO i = 1, m
                    c(i, n-k+j) = c(i, n-k+j) - work(i, j)
                  END DO
                END DO
              END IF
            END IF
          ELSE IF (LSAME(storev,'R')) THEN
            IF (LSAME(direct,'F')) THEN
              IF (LSAME(side,'L')) THEN
                DO j = 1, k
                  CALL DCOPY(n, c(j,1), ldc, work(1,j), 1_ip_)
                END DO
                CALL DTRMM('Right', 'Upper', 'Transpose', 'Unit', n, k,     &
                  one, v, ldv, work, ldwork)
                IF (m>k) THEN
                  CALL DGEMM('Transpose', 'Transpose', n, k, m-k, one,      &
                    c(k+1,1), ldc, v(1,k+1), ldv, one, work, ldwork)
                END IF
                CALL DTRMM('Right', 'Upper', transt, 'Non-unit', n, k, one, &
                  t, ldt, work, ldwork)
                IF (m>k) THEN
                  CALL DGEMM('Transpose', 'Transpose', m-k, n, k, -one,     &
                    v(1,k+1), ldv, work, ldwork, one, c(k+1,1), ldc)
                END IF
                CALL DTRMM('Right', 'Upper', 'No transpose', 'Unit', n, k,  &
                  one, v, ldv, work, ldwork)
                DO j = 1, k
                  DO i = 1, n
                    c(j, i) = c(j, i) - work(i, j)
                  END DO
                END DO
              ELSE IF (LSAME(side,'R')) THEN
                DO j = 1, k
                  CALL DCOPY(m, c(1,j), 1_ip_, work(1,j), 1_ip_)
                END DO
                CALL DTRMM('Right', 'Upper', 'Transpose', 'Unit', m, k,     &
                  one, v, ldv, work, ldwork)
                IF (n>k) THEN
                  CALL DGEMM('No transpose', 'Transpose', m, k, n-k, one,   &
                    c(1,k+1), ldc, v(1,k+1), ldv, one, work, ldwork)
                END IF
                CALL DTRMM('Right', 'Upper', trans, 'Non-unit', m, k, one,  &
                  t, ldt, work, ldwork)
                IF (n>k) THEN
                  CALL DGEMM('No transpose', 'No transpose', m, n-k, k,     &
                    -one, work, ldwork, v(1,k+1), ldv, one, c(1,k+1), ldc)
                END IF
                CALL DTRMM('Right', 'Upper', 'No transpose', 'Unit', m, k,  &
                  one, v, ldv, work, ldwork)
                DO j = 1, k
                  DO i = 1, m
                    c(i, j) = c(i, j) - work(i, j)
                  END DO
                END DO
              END IF
            ELSE
              IF (LSAME(side,'L')) THEN
                DO j = 1, k
                  CALL DCOPY(n, c(m-k+j,1), ldc, work(1,j), 1_ip_)
                END DO
                CALL DTRMM('Right', 'Lower', 'Transpose', 'Unit', n, k,     &
                  one, v(1,m-k+1), ldv, work, ldwork)
                IF (m>k) THEN
                  CALL DGEMM('Transpose', 'Transpose', n, k, m-k, one, c,   &
                    ldc, v, ldv, one, work, ldwork)
                END IF
                CALL DTRMM('Right', 'Lower', transt, 'Non-unit', n, k, one, &
                  t, ldt, work, ldwork)
                IF (m>k) THEN
                  CALL DGEMM('Transpose', 'Transpose', m-k, n, k, -one, v,  &
                    ldv, work, ldwork, one, c, ldc)
                END IF
                CALL DTRMM('Right', 'Lower', 'No transpose', 'Unit', n, k,  &
                  one, v(1,m-k+1), ldv, work, ldwork)
                DO j = 1, k
                  DO i = 1, n
                    c(m-k+j, i) = c(m-k+j, i) - work(i, j)
                  END DO
                END DO
              ELSE IF (LSAME(side,'R')) THEN
                DO j = 1, k
                  CALL DCOPY(m, c(1,n-k+j), 1_ip_, work(1,j), 1_ip_)
                END DO
                CALL DTRMM('Right', 'Lower', 'Transpose', 'Unit', m, k,     &
                  one, v(1,n-k+1), ldv, work, ldwork)
                IF (n>k) THEN
                  CALL DGEMM('No transpose', 'Transpose', m, k, n-k, one,   &
                    c, ldc, v, ldv, one, work, ldwork)
                END IF
                CALL DTRMM('Right', 'Lower', trans, 'Non-unit', m, k, one,  &
                  t, ldt, work, ldwork)
                IF (n>k) THEN
                  CALL DGEMM('No transpose', 'No transpose', m, n-k, k,     &
                    -one, work, ldwork, v, ldv, one, c, ldc)
                END IF
                CALL DTRMM('Right', 'Lower', 'No transpose', 'Unit', m, k,  &
                  one, v(1,n-k+1), ldv, work, ldwork)
                DO j = 1, k
                  DO i = 1, m
                    c(i, n-k+j) = c(i, n-k+j) - work(i, j)
                  END DO
                END DO
              END IF
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLARFG(n, alpha, x, incx, tau)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: incx, n
          REAL(rp_) :: alpha, tau
          REAL(rp_) :: x(*)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          INTEGER(ip_) :: j, knt
          REAL(rp_) :: beta, rsafmn, safmin, xnorm
          REAL(rp_) :: DLAMCH, DLAPY2, DNRM2
          EXTERNAL :: DLAMCH, DLAPY2, DNRM2
          INTRINSIC :: ABS, SIGN
          EXTERNAL :: DSCAL
          IF (n<=1) THEN
            tau = zero
            RETURN
          END IF
          xnorm = DNRM2(n-1, x, incx)
          IF (xnorm==zero) THEN
            tau = zero
          ELSE
            beta = -SIGN(DLAPY2(alpha,xnorm), alpha)
            safmin = DLAMCH('S')/DLAMCH('E')
            knt = 0
            IF (ABS(beta)<safmin) THEN
              rsafmn = one/safmin
 10           CONTINUE
              knt = knt + 1
              CALL DSCAL(n-1, rsafmn, x, incx)
              beta = beta*rsafmn
              alpha = alpha*rsafmn
              IF ((ABS(beta)<safmin) .AND. (knt<20)) GO TO 10
              xnorm = DNRM2(n-1, x, incx)
              beta = -SIGN(DLAPY2(alpha,xnorm), alpha)
            END IF
            tau = (beta-alpha)/beta
            CALL DSCAL(n-1, one/(alpha-beta), x, incx)
            DO j = 1, knt
              beta = beta*safmin
            END DO
            alpha = beta
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLARFT(direct, storev, n, k, v, ldv, tau, t, ldt)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: direct, storev
          INTEGER(ip_) :: k, ldt, ldv, n
          REAL(rp_) :: t(ldt, *), tau(*), v(ldv, *)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          INTEGER(ip_) :: i, j, prevlastv, lastv
          EXTERNAL :: DGEMV, DTRMV
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          IF (n==0) RETURN
          IF (LSAME(direct,'F')) THEN
            prevlastv = n
            DO i = 1, k
              prevlastv = MAX(i, prevlastv)
              IF (tau(i)==zero) THEN
                DO j = 1, i
                  t(j, i) = zero
                END DO
              ELSE
                IF (LSAME(storev,'C')) THEN
                  DO lastv = n, i + 1, -1_ip_
                    IF (v(lastv,i)/=zero) EXIT
                  END DO
                  DO j = 1, i - 1
                    t(j, i) = -tau(i)*v(i, j)
                  END DO
                  j = MIN(lastv, prevlastv)
                  CALL DGEMV('Transpose', j-i, i-1, -tau(i), v(i+1,1), ldv, &
                    v(i+1,i), 1_ip_, one, t(1,i), 1_ip_)
                ELSE
                  DO lastv = n, i + 1, -1_ip_
                    IF (v(i,lastv)/=zero) EXIT
                  END DO
                  DO j = 1, i - 1
                    t(j, i) = -tau(i)*v(j, i)
                  END DO
                  j = MIN(lastv, prevlastv)
                  CALL DGEMV('No transpose', i-1, j-i, -tau(i), v(1,i+1),   &
                    ldv, v(i,i+1), ldv, one, t(1,i), 1_ip_)
                END IF
                CALL DTRMV('Upper', 'No transpose', 'Non-unit', i-1, t,     &
                  ldt, t(1,i), 1_ip_)
                t(i, i) = tau(i)
                IF (i>1) THEN
                  prevlastv = MAX(prevlastv, lastv)
                ELSE
                  prevlastv = lastv
                END IF
              END IF
            END DO
          ELSE
            prevlastv = 1
            DO i = k, 1_ip_, -1_ip_
              IF (tau(i)==zero) THEN
                DO j = i, k
                  t(j, i) = zero
                END DO
              ELSE
                IF (i<k) THEN
                  IF (LSAME(storev,'C')) THEN
                    DO lastv = 1, i - 1
                      IF (v(lastv,i)/=zero) EXIT
                    END DO
                    DO j = i + 1, k
                      t(j, i) = -tau(i)*v(n-k+i, j)
                    END DO
                    j = MAX(lastv, prevlastv)
                    CALL DGEMV('Transpose', n-k+i-j, k-i, -tau(i), v(j,     &
                      i+1), ldv, v(j,i), 1_ip_, one, t(i+1,i), 1_ip_)
                  ELSE
                    DO lastv = 1, i - 1
                      IF (v(i,lastv)/=zero) EXIT
                    END DO
                    DO j = i + 1, k
                      t(j, i) = -tau(i)*v(j, n-k+i)
                    END DO
                    j = MAX(lastv, prevlastv)
                    CALL DGEMV('No transpose', k-i, n-k+i-j, -tau(i),       &
                      v(i+1,j), ldv, v(i,j), ldv, one, t(i+1,i), 1_ip_)
                  END IF
                  CALL DTRMV('Lower', 'No transpose', 'Non-unit', k-i,      &
                    t(i+1,i+1), ldt, t(i+1,i), 1_ip_)
                  IF (i>1) THEN
                    prevlastv = MIN(prevlastv, lastv)
                  ELSE
                    prevlastv = lastv
                  END IF
                END IF
                t(i, i) = tau(i)
              END IF
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLARFX(side, m, n, v, tau, c, ldc, work)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: side
          INTEGER(ip_) :: ldc, m, n
          REAL(rp_) :: tau
          REAL(rp_) :: c(ldc, *), v(*), work(*)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          INTEGER(ip_) :: j
          REAL(rp_) :: sum, t1, t10, t2, t3, t4, t5, t6, t7, t8, t9, v1,    &
            v10, v2, v3, v4, v5, v6, v7, v8, v9
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: DLARF
          IF (tau==zero) RETURN
          IF (LSAME(side,'L')) THEN
            GO TO (10, 30, 50, 70, 90, 110, 130, 150, 170, 190) m
            CALL DLARF(side, m, n, v, 1_ip_, tau, c, ldc, work)
            GO TO 410
 10         CONTINUE
            t1 = one - tau*v(1)*v(1)
            DO j = 1, n
              c(1, j) = t1*c(1, j)
            END DO
            GO TO 410
 30         CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            DO j = 1, n
              sum = v1*c(1, j) + v2*c( 2_ip_, j)
              c(1, j) = c(1, j) - sum*t1
              c( 2_ip_, j) = c( 2_ip_, j) - sum*t2
            END DO
            GO TO 410
 50         CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            v3 = v(3)
            t3 = tau*v3
            DO j = 1, n
              sum = v1*c(1, j) + v2*c( 2_ip_, j) + v3*c( 3_ip_, j)
              c(1, j) = c(1, j) - sum*t1
              c( 2_ip_, j) = c( 2_ip_, j) - sum*t2
              c( 3_ip_, j) = c( 3_ip_, j) - sum*t3
            END DO
            GO TO 410
 70         CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            v3 = v(3)
            t3 = tau*v3
            v4 = v(4)
            t4 = tau*v4
            DO j = 1, n
              sum = v1*c(1, j) + v2*c( 2_ip_, j) + v3*c( 3_ip_, j) + v4*c(  &
                4_ip_, j)
              c(1, j) = c(1, j) - sum*t1
              c( 2_ip_, j) = c( 2_ip_, j) - sum*t2
              c( 3_ip_, j) = c( 3_ip_, j) - sum*t3
              c( 4_ip_, j) = c( 4_ip_, j) - sum*t4
            END DO
            GO TO 410
 90         CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            v3 = v(3)
            t3 = tau*v3
            v4 = v(4)
            t4 = tau*v4
            v5 = v(5)
            t5 = tau*v5
            DO j = 1, n
              sum = v1*c(1, j) + v2*c( 2_ip_, j) + v3*c( 3_ip_, j) + v4*c(  &
                4_ip_, j) + v5*c(5, j)
              c(1, j) = c(1, j) - sum*t1
              c( 2_ip_, j) = c( 2_ip_, j) - sum*t2
              c( 3_ip_, j) = c( 3_ip_, j) - sum*t3
              c( 4_ip_, j) = c( 4_ip_, j) - sum*t4
              c(5, j) = c(5, j) - sum*t5
            END DO
            GO TO 410
 110        CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            v3 = v(3)
            t3 = tau*v3
            v4 = v(4)
            t4 = tau*v4
            v5 = v(5)
            t5 = tau*v5
            v6 = v(6)
            t6 = tau*v6
            DO j = 1, n
              sum = v1*c(1, j) + v2*c( 2_ip_, j) + v3*c( 3_ip_, j) + v4*c(  &
                4_ip_, j) + v5*c(5, j) + v6*c(6, j)
              c(1, j) = c(1, j) - sum*t1
              c( 2_ip_, j) = c( 2_ip_, j) - sum*t2
              c( 3_ip_, j) = c( 3_ip_, j) - sum*t3
              c( 4_ip_, j) = c( 4_ip_, j) - sum*t4
              c(5, j) = c(5, j) - sum*t5
              c(6, j) = c(6, j) - sum*t6
            END DO
            GO TO 410
 130        CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            v3 = v(3)
            t3 = tau*v3
            v4 = v(4)
            t4 = tau*v4
            v5 = v(5)
            t5 = tau*v5
            v6 = v(6)
            t6 = tau*v6
            v7 = v(7)
            t7 = tau*v7
            DO j = 1, n
              sum = v1*c(1, j) + v2*c( 2_ip_, j) + v3*c( 3_ip_, j) + v4*c(  &
                4_ip_, j) + v5*c(5, j) + v6*c(6, j) + v7*c(7, j)
              c(1, j) = c(1, j) - sum*t1
              c( 2_ip_, j) = c( 2_ip_, j) - sum*t2
              c( 3_ip_, j) = c( 3_ip_, j) - sum*t3
              c( 4_ip_, j) = c( 4_ip_, j) - sum*t4
              c(5, j) = c(5, j) - sum*t5
              c(6, j) = c(6, j) - sum*t6
              c(7, j) = c(7, j) - sum*t7
            END DO
            GO TO 410
 150        CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            v3 = v(3)
            t3 = tau*v3
            v4 = v(4)
            t4 = tau*v4
            v5 = v(5)
            t5 = tau*v5
            v6 = v(6)
            t6 = tau*v6
            v7 = v(7)
            t7 = tau*v7
            v8 = v(8)
            t8 = tau*v8
            DO j = 1, n
              sum = v1*c(1, j) + v2*c( 2_ip_, j) + v3*c( 3_ip_, j) + v4*c(  &
                4_ip_, j) + v5*c(5, j) + v6*c(6, j) + v7*c(7, j) + v8*c(8, j)
              c(1, j) = c(1, j) - sum*t1
              c( 2_ip_, j) = c( 2_ip_, j) - sum*t2
              c( 3_ip_, j) = c( 3_ip_, j) - sum*t3
              c( 4_ip_, j) = c( 4_ip_, j) - sum*t4
              c(5, j) = c(5, j) - sum*t5
              c(6, j) = c(6, j) - sum*t6
              c(7, j) = c(7, j) - sum*t7
              c(8, j) = c(8, j) - sum*t8
            END DO
            GO TO 410
 170        CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            v3 = v(3)
            t3 = tau*v3
            v4 = v(4)
            t4 = tau*v4
            v5 = v(5)
            t5 = tau*v5
            v6 = v(6)
            t6 = tau*v6
            v7 = v(7)
            t7 = tau*v7
            v8 = v(8)
            t8 = tau*v8
            v9 = v(9)
            t9 = tau*v9
            DO j = 1, n
              sum = v1*c(1, j) + v2*c( 2_ip_, j) + v3*c( 3_ip_, j) + v4*c(  &
                4_ip_, j) + v5*c(5, j) + v6*c(6, j) + v7*c(7, j) + v8*c(8,  &
                j) + v9*c(9, j)
              c(1, j) = c(1, j) - sum*t1
              c( 2_ip_, j) = c( 2_ip_, j) - sum*t2
              c( 3_ip_, j) = c( 3_ip_, j) - sum*t3
              c( 4_ip_, j) = c( 4_ip_, j) - sum*t4
              c(5, j) = c(5, j) - sum*t5
              c(6, j) = c(6, j) - sum*t6
              c(7, j) = c(7, j) - sum*t7
              c(8, j) = c(8, j) - sum*t8
              c(9, j) = c(9, j) - sum*t9
            END DO
            GO TO 410
 190        CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            v3 = v(3)
            t3 = tau*v3
            v4 = v(4)
            t4 = tau*v4
            v5 = v(5)
            t5 = tau*v5
            v6 = v(6)
            t6 = tau*v6
            v7 = v(7)
            t7 = tau*v7
            v8 = v(8)
            t8 = tau*v8
            v9 = v(9)
            t9 = tau*v9
            v10 = v(10)
            t10 = tau*v10
            DO j = 1, n
              sum = v1*c(1, j) + v2*c( 2_ip_, j) + v3*c( 3_ip_, j) + v4*c(  &
                4_ip_, j) + v5*c(5, j) + v6*c(6, j) + v7*c(7, j) + v8*c(8,  &
                j) + v9*c(9, j) + v10*c(10, j)
              c(1, j) = c(1, j) - sum*t1
              c( 2_ip_, j) = c( 2_ip_, j) - sum*t2
              c( 3_ip_, j) = c( 3_ip_, j) - sum*t3
              c( 4_ip_, j) = c( 4_ip_, j) - sum*t4
              c(5, j) = c(5, j) - sum*t5
              c(6, j) = c(6, j) - sum*t6
              c(7, j) = c(7, j) - sum*t7
              c(8, j) = c(8, j) - sum*t8
              c(9, j) = c(9, j) - sum*t9
              c(10, j) = c(10, j) - sum*t10
            END DO
            GO TO 410
          ELSE
            GO TO (210, 230, 250, 270, 290, 310, 330, 350, 370, 390) n
            CALL DLARF(side, m, n, v, 1_ip_, tau, c, ldc, work)
            GO TO 410
 210        CONTINUE
            t1 = one - tau*v(1)*v(1)
            DO j = 1, m
              c(j, 1_ip_) = t1*c(j, 1_ip_)
            END DO
            GO TO 410
 230        CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            DO j = 1, m
              sum = v1*c(j, 1_ip_) + v2*c(j, 2_ip_)
              c(j, 1_ip_) = c(j, 1_ip_) - sum*t1
              c(j, 2_ip_) = c(j, 2_ip_) - sum*t2
            END DO
            GO TO 410
 250        CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            v3 = v(3)
            t3 = tau*v3
            DO j = 1, m
              sum = v1*c(j, 1_ip_) + v2*c(j, 2_ip_) + v3*c(j, 3_ip_)
              c(j, 1_ip_) = c(j, 1_ip_) - sum*t1
              c(j, 2_ip_) = c(j, 2_ip_) - sum*t2
              c(j, 3_ip_) = c(j, 3_ip_) - sum*t3
            END DO
            GO TO 410
 270        CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            v3 = v(3)
            t3 = tau*v3
            v4 = v(4)
            t4 = tau*v4
            DO j = 1, m
              sum = v1*c(j, 1_ip_) + v2*c(j, 2_ip_) + v3*c(j, 3_ip_) +      &
                v4*c(j, 4_ip_)
              c(j, 1_ip_) = c(j, 1_ip_) - sum*t1
              c(j, 2_ip_) = c(j, 2_ip_) - sum*t2
              c(j, 3_ip_) = c(j, 3_ip_) - sum*t3
              c(j, 4_ip_) = c(j, 4_ip_) - sum*t4
            END DO
            GO TO 410
 290        CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            v3 = v(3)
            t3 = tau*v3
            v4 = v(4)
            t4 = tau*v4
            v5 = v(5)
            t5 = tau*v5
            DO j = 1, m
              sum = v1*c(j, 1_ip_) + v2*c(j, 2_ip_) + v3*c(j, 3_ip_) +      &
                v4*c(j, 4_ip_) + v5*c(j, 5)
              c(j, 1_ip_) = c(j, 1_ip_) - sum*t1
              c(j, 2_ip_) = c(j, 2_ip_) - sum*t2
              c(j, 3_ip_) = c(j, 3_ip_) - sum*t3
              c(j, 4_ip_) = c(j, 4_ip_) - sum*t4
              c(j, 5) = c(j, 5) - sum*t5
            END DO
            GO TO 410
 310        CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            v3 = v(3)
            t3 = tau*v3
            v4 = v(4)
            t4 = tau*v4
            v5 = v(5)
            t5 = tau*v5
            v6 = v(6)
            t6 = tau*v6
            DO j = 1, m
              sum = v1*c(j, 1_ip_) + v2*c(j, 2_ip_) + v3*c(j, 3_ip_) +      &
                v4*c(j, 4_ip_) + v5*c(j, 5) + v6*c(j, 6)
              c(j, 1_ip_) = c(j, 1_ip_) - sum*t1
              c(j, 2_ip_) = c(j, 2_ip_) - sum*t2
              c(j, 3_ip_) = c(j, 3_ip_) - sum*t3
              c(j, 4_ip_) = c(j, 4_ip_) - sum*t4
              c(j, 5) = c(j, 5) - sum*t5
              c(j, 6) = c(j, 6) - sum*t6
            END DO
            GO TO 410
 330        CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            v3 = v(3)
            t3 = tau*v3
            v4 = v(4)
            t4 = tau*v4
            v5 = v(5)
            t5 = tau*v5
            v6 = v(6)
            t6 = tau*v6
            v7 = v(7)
            t7 = tau*v7
            DO j = 1, m
              sum = v1*c(j, 1_ip_) + v2*c(j, 2_ip_) + v3*c(j, 3_ip_) +      &
                v4*c(j, 4_ip_) + v5*c(j, 5) + v6*c(j, 6) + v7*c(j, 7)
              c(j, 1_ip_) = c(j, 1_ip_) - sum*t1
              c(j, 2_ip_) = c(j, 2_ip_) - sum*t2
              c(j, 3_ip_) = c(j, 3_ip_) - sum*t3
              c(j, 4_ip_) = c(j, 4_ip_) - sum*t4
              c(j, 5) = c(j, 5) - sum*t5
              c(j, 6) = c(j, 6) - sum*t6
              c(j, 7) = c(j, 7) - sum*t7
            END DO
            GO TO 410
 350        CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            v3 = v(3)
            t3 = tau*v3
            v4 = v(4)
            t4 = tau*v4
            v5 = v(5)
            t5 = tau*v5
            v6 = v(6)
            t6 = tau*v6
            v7 = v(7)
            t7 = tau*v7
            v8 = v(8)
            t8 = tau*v8
            DO j = 1, m
              sum = v1*c(j, 1_ip_) + v2*c(j, 2_ip_) + v3*c(j, 3_ip_) +      &
                v4*c(j, 4_ip_) + v5*c(j, 5) + v6*c(j, 6) + v7*c(j, 7) +     &
                v8*c(j, 8)
              c(j, 1_ip_) = c(j, 1_ip_) - sum*t1
              c(j, 2_ip_) = c(j, 2_ip_) - sum*t2
              c(j, 3_ip_) = c(j, 3_ip_) - sum*t3
              c(j, 4_ip_) = c(j, 4_ip_) - sum*t4
              c(j, 5) = c(j, 5) - sum*t5
              c(j, 6) = c(j, 6) - sum*t6
              c(j, 7) = c(j, 7) - sum*t7
              c(j, 8) = c(j, 8) - sum*t8
            END DO
            GO TO 410
 370        CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            v3 = v(3)
            t3 = tau*v3
            v4 = v(4)
            t4 = tau*v4
            v5 = v(5)
            t5 = tau*v5
            v6 = v(6)
            t6 = tau*v6
            v7 = v(7)
            t7 = tau*v7
            v8 = v(8)
            t8 = tau*v8
            v9 = v(9)
            t9 = tau*v9
            DO j = 1, m
              sum = v1*c(j, 1_ip_) + v2*c(j, 2_ip_) + v3*c(j, 3_ip_) +      &
                v4*c(j, 4_ip_) + v5*c(j, 5) + v6*c(j, 6) + v7*c(j, 7) +     &
                v8*c(j, 8) + v9*c(j, 9)
              c(j, 1_ip_) = c(j, 1_ip_) - sum*t1
              c(j, 2_ip_) = c(j, 2_ip_) - sum*t2
              c(j, 3_ip_) = c(j, 3_ip_) - sum*t3
              c(j, 4_ip_) = c(j, 4_ip_) - sum*t4
              c(j, 5) = c(j, 5) - sum*t5
              c(j, 6) = c(j, 6) - sum*t6
              c(j, 7) = c(j, 7) - sum*t7
              c(j, 8) = c(j, 8) - sum*t8
              c(j, 9) = c(j, 9) - sum*t9
            END DO
            GO TO 410
 390        CONTINUE
            v1 = v(1)
            t1 = tau*v1
            v2 = v(2)
            t2 = tau*v2
            v3 = v(3)
            t3 = tau*v3
            v4 = v(4)
            t4 = tau*v4
            v5 = v(5)
            t5 = tau*v5
            v6 = v(6)
            t6 = tau*v6
            v7 = v(7)
            t7 = tau*v7
            v8 = v(8)
            t8 = tau*v8
            v9 = v(9)
            t9 = tau*v9
            v10 = v(10)
            t10 = tau*v10
            DO j = 1, m
              sum = v1*c(j, 1_ip_) + v2*c(j, 2_ip_) + v3*c(j, 3_ip_) +      &
                v4*c(j, 4_ip_) + v5*c(j, 5) + v6*c(j, 6) + v7*c(j, 7) +     &
                v8*c(j, 8) + v9*c(j, 9) + v10*c(j, 10)
              c(j, 1_ip_) = c(j, 1_ip_) - sum*t1
              c(j, 2_ip_) = c(j, 2_ip_) - sum*t2
              c(j, 3_ip_) = c(j, 3_ip_) - sum*t3
              c(j, 4_ip_) = c(j, 4_ip_) - sum*t4
              c(j, 5) = c(j, 5) - sum*t5
              c(j, 6) = c(j, 6) - sum*t6
              c(j, 7) = c(j, 7) - sum*t7
              c(j, 8) = c(j, 8) - sum*t8
              c(j, 9) = c(j, 9) - sum*t9
              c(j, 10) = c(j, 10) - sum*t10
            END DO
            GO TO 410
          END IF
 410      CONTINUE
          RETURN
        END SUBROUTINE

        SUBROUTINE DLARTG(f, g, cs, sn, r)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: cs, f, g, r, sn
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          REAL(rp_) :: two
          PARAMETER (two=2.0_rp_)
          INTEGER(ip_) :: count, i
          REAL(rp_) :: eps, f1, g1, safmin, safmn2, safmx2, scale
          REAL(rp_) :: DLAMCH
          EXTERNAL :: DLAMCH
          INTRINSIC :: ABS, INT, LOG, MAX, SQRT
          safmin = DLAMCH('S')
          eps = DLAMCH('E')
          safmn2 = DLAMCH('B')**INT(LOG(safmin/eps)/LOG(DLAMCH('B'))/two)
          safmx2 = one/safmn2
          IF (g==zero) THEN
            cs = one
            sn = zero
            r = f
          ELSE IF (f==zero) THEN
            cs = zero
            sn = one
            r = g
          ELSE
            f1 = f
            g1 = g
            scale = MAX(ABS(f1), ABS(g1))
            IF (scale>=safmx2) THEN
              count = 0
 10           CONTINUE
              count = count + 1
              f1 = f1*safmn2
              g1 = g1*safmn2
              scale = MAX(ABS(f1), ABS(g1))
              IF (scale>=safmx2 .AND. count<20) GO TO 10
              r = SQRT(f1**2+g1**2)
              cs = f1/r
              sn = g1/r
              DO i = 1, count
                r = r*safmx2
              END DO
            ELSE IF (scale<=safmn2) THEN
              count = 0
 30           CONTINUE
              count = count + 1
              f1 = f1*safmx2
              g1 = g1*safmx2
              scale = MAX(ABS(f1), ABS(g1))
              IF (scale<=safmn2) GO TO 30
              r = SQRT(f1**2+g1**2)
              cs = f1/r
              sn = g1/r
              DO i = 1, count
                r = r*safmn2
              END DO
            ELSE
              r = SQRT(f1**2+g1**2)
              cs = f1/r
              sn = g1/r
            END IF
            IF (ABS(f)>ABS(g) .AND. cs<zero) THEN
              cs = -cs
              sn = -sn
              r = -r
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLARZ(side, m, n, l, v, incv, tau, c, ldc, work)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: side
          INTEGER(ip_) :: incv, l, ldc, m, n
          REAL(rp_) :: tau
          REAL(rp_) :: c(ldc, *), v(*), work(*)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          EXTERNAL :: DAXPY, DCOPY, DGEMV, DGER
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          IF (LSAME(side,'L')) THEN
            IF (tau/=zero) THEN
              CALL DCOPY(n, c, ldc, work, 1_ip_)
              CALL DGEMV('Transpose', l, n, one, c(m-l+1,1), ldc, v, incv,  &
                one, work, 1_ip_)
              CALL DAXPY(n, -tau, work, 1_ip_, c, ldc)
              CALL DGER(l, n, -tau, v, incv, work, 1_ip_, c(m-l+1,1), ldc)
            END IF
          ELSE
            IF (tau/=zero) THEN
              CALL DCOPY(m, c, 1_ip_, work, 1_ip_)
              CALL DGEMV('No transpose', m, l, one, c(1,n-l+1), ldc, v,     &
                incv, one, work, 1_ip_)
              CALL DAXPY(m, -tau, work, 1_ip_, c, 1_ip_)
              CALL DGER(m, l, -tau, work, 1_ip_, v, incv, c(1,n-l+1), ldc)
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLARZB(side, trans, direct, storev, m, n, k, l, v, ldv,  &
          t, ldt, c, ldc, work, ldwork)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: direct, side, storev, trans
          INTEGER(ip_) :: k, l, ldc, ldt, ldv, ldwork, m, n
          REAL(rp_) :: c(ldc, *), t(ldt, *), v(ldv, *), work(ldwork, *)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          CHARACTER :: transt
          INTEGER(ip_) :: i, info, j
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: DCOPY, DGEMM, DTRMM, XERBLA2
          IF (m<=0 .OR. n<=0) RETURN
          info = 0
          IF (.NOT. LSAME(direct,'B')) THEN
            info = -3
          ELSE IF (.NOT. LSAME(storev,'R')) THEN
            info = -4
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('LARZB', -info)
            RETURN
          END IF
          IF (LSAME(trans,'N')) THEN
            transt = 'T'
          ELSE
            transt = 'N'
          END IF
          IF (LSAME(side,'L')) THEN
            DO j = 1, k
              CALL DCOPY(n, c(j,1), ldc, work(1,j), 1_ip_)
            END DO
            IF (l>0) CALL DGEMM('Transpose', 'Transpose', n, k, l, one,     &
              c(m-l+1,1), ldc, v, ldv, one, work, ldwork)
            CALL DTRMM('Right', 'Lower', transt, 'Non-unit', n, k, one, t,  &
              ldt, work, ldwork)
            DO j = 1, n
              DO i = 1, k
                c(i, j) = c(i, j) - work(j, i)
              END DO
            END DO
            IF (l>0) CALL DGEMM('Transpose', 'Transpose', l, n, k, -one, v, &
              ldv, work, ldwork, one, c(m-l+1,1), ldc)
          ELSE IF (LSAME(side,'R')) THEN
            DO j = 1, k
              CALL DCOPY(m, c(1,j), 1_ip_, work(1,j), 1_ip_)
            END DO
            IF (l>0) CALL DGEMM('No transpose', 'Transpose', m, k, l, one,  &
              c(1,n-l+1), ldc, v, ldv, one, work, ldwork)
            CALL DTRMM('Right', 'Lower', trans, 'Non-unit', m, k, one, t,   &
              ldt, work, ldwork)
            DO j = 1, k
              DO i = 1, m
                c(i, j) = c(i, j) - work(i, j)
              END DO
            END DO
            IF (l>0) CALL DGEMM('No transpose', 'No transpose', m, l, k,    &
              -one, work, ldwork, v, ldv, one, c(1,n-l+1), ldc)
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLARZT(direct, storev, n, k, v, ldv, tau, t, ldt)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: direct, storev
          INTEGER(ip_) :: k, ldt, ldv, n
          REAL(rp_) :: t(ldt, *), tau(*), v(ldv, *)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          INTEGER(ip_) :: i, info, j
          EXTERNAL :: DGEMV, DTRMV, XERBLA2
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          info = 0
          IF (.NOT. LSAME(direct,'B')) THEN
            info = -1
          ELSE IF (.NOT. LSAME(storev,'R')) THEN
            info = -2
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('LARZT', -info)
            RETURN
          END IF
          DO i = k, 1_ip_, -1_ip_
            IF (tau(i)==zero) THEN
              DO j = i, k
                t(j, i) = zero
              END DO
            ELSE
              IF (i<k) THEN
                CALL DGEMV('No transpose', k-i, n, -tau(i), v(i+1,1), ldv,  &
                  v(i,1), ldv, zero, t(i+1,i), 1_ip_)
                CALL DTRMV('Lower', 'No transpose', 'Non-unit', k-i, t(i+1, &
                  i+1), ldt, t(i+1,i), 1_ip_)
              END IF
              t(i, i) = tau(i)
            END IF
          END DO
          RETURN
        END SUBROUTINE

        SUBROUTINE DLAS2(f, g, h, ssmin, ssmax)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: f, g, h, ssmax, ssmin
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          REAL(rp_) :: two
          PARAMETER (two=2.0_rp_)
          REAL(rp_) :: as, at, au, c, fa, fhmn, fhmx, ga, ha
          INTRINSIC :: ABS, MAX, MIN, SQRT
          fa = ABS(f)
          ga = ABS(g)
          ha = ABS(h)
          fhmn = MIN(fa, ha)
          fhmx = MAX(fa, ha)
          IF (fhmn==zero) THEN
            ssmin = zero
            IF (fhmx==zero) THEN
              ssmax = ga
            ELSE
              ssmax = MAX(fhmx, ga)*SQRT(one+(MIN(fhmx,ga)/MAX(fhmx,        &
                ga))**2)
            END IF
          ELSE
            IF (ga<fhmx) THEN
              as = one + fhmn/fhmx
              at = (fhmx-fhmn)/fhmx
              au = (ga/fhmx)**2
              c = two/(SQRT(as*as+au)+SQRT(at*at+au))
              ssmin = fhmn*c
              ssmax = fhmx/c
            ELSE
              au = fhmx/ga
              IF (au==zero) THEN
                ssmin = (fhmn*fhmx)/ga
                ssmax = ga
              ELSE
                as = one + fhmn/fhmx
                at = (fhmx-fhmn)/fhmx
                c = one/(SQRT(one+(as*au)**2)+SQRT(one+(at*au)**2))
                ssmin = (fhmn*c)*au
                ssmin = ssmin + ssmin
                ssmax = ga/(c+c)
              END IF
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASCL(type, kl, ku, cfrom, cto, m, n, a, lda, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: type
          INTEGER(ip_) :: info, kl, ku, lda, m, n
          REAL(rp_) :: cfrom, cto
          REAL(rp_) :: a(lda, *)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          LOGICAL :: done
          INTEGER(ip_) :: i, itype, j, k1, k2, k3, k4
          REAL(rp_) :: bignum, cfrom1, cfromc, cto1, ctoc, mul, smlnum
          LOGICAL :: LISNAN_cfrom, LISNAN_cto
          LOGICAL :: LSAME, DISNAN
          REAL(rp_) :: DLAMCH
          EXTERNAL :: LSAME, DLAMCH, DISNAN
          INTRINSIC :: ABS, MAX, MIN
          EXTERNAL :: XERBLA2
          info = 0
          IF (LSAME(type,'G')) THEN
            itype = 0
          ELSE IF (LSAME(type,'L')) THEN
            itype = 1
          ELSE IF (LSAME(type,'U')) THEN
            itype = 2
          ELSE IF (LSAME(type,'H')) THEN
            itype = 3
          ELSE IF (LSAME(type,'B')) THEN
            itype = 4
          ELSE IF (LSAME(type,'Q')) THEN
            itype = 5
          ELSE IF (LSAME(type,'Z')) THEN
            itype = 6
          ELSE
            itype = -1
          END IF
          LISNAN_cfrom = DISNAN(cfrom)
          LISNAN_cto = DISNAN(cto)
          IF (itype==-1) THEN
            info = -1
          ELSE IF (cfrom==zero .OR. LISNAN_cfrom) THEN
            info = -4
          ELSE IF (LISNAN_cto) THEN
            info = -5
          ELSE IF (m<0) THEN
            info = -6
          ELSE IF (n<0 .OR. (itype==4 .AND. n/=m) .OR. (itype==5 .AND.      &
            n/=m)) THEN
            info = -7
          ELSE IF (itype<=3 .AND. lda<MAX(1,m)) THEN
            info = -9
          ELSE IF (itype>=4) THEN
            IF (kl<0 .OR. kl>MAX(m-1,0)) THEN
              info = -2
            ELSE IF (ku<0 .OR. ku>MAX(n-1,0) .OR. ((itype==4 .OR. itype==   &
              5) .AND. kl/=ku)) THEN
              info = -3
            ELSE IF ((itype==4 .AND. lda<kl+1) .OR. (itype==5 .AND.         &
              lda<ku+1) .OR. (itype==6 .AND. lda<2*kl+ku +1)) THEN
              info = -9
            END IF
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('LASCL', -info)
            RETURN
          END IF
          IF (n==0 .OR. m==0) RETURN
          smlnum = DLAMCH('S')
          bignum = one/smlnum
          cfromc = cfrom
          ctoc = cto
 10       CONTINUE
          cfrom1 = cfromc*smlnum
          IF (cfrom1==cfromc) THEN
            mul = ctoc/cfromc
            done = .TRUE.
            cto1 = ctoc
          ELSE
            cto1 = ctoc/bignum
            IF (cto1==ctoc) THEN
              mul = ctoc
              done = .TRUE.
              cfromc = one
            ELSE IF (ABS(cfrom1)>ABS(ctoc) .AND. ctoc/=zero) THEN
              mul = smlnum
              done = .FALSE.
              cfromc = cfrom1
            ELSE IF (ABS(cto1)>ABS(cfromc)) THEN
              mul = bignum
              done = .FALSE.
              ctoc = cto1
            ELSE
              mul = ctoc/cfromc
              done = .TRUE.
            END IF
          END IF
          IF (itype==0) THEN
            DO j = 1, n
              DO i = 1, m
                a(i, j) = a(i, j)*mul
              END DO
            END DO
          ELSE IF (itype==1) THEN
            DO j = 1, n
              DO i = j, m
                a(i, j) = a(i, j)*mul
              END DO
            END DO
          ELSE IF (itype==2) THEN
            DO j = 1, n
              DO i = 1, MIN(j, m)
                a(i, j) = a(i, j)*mul
              END DO
            END DO
          ELSE IF (itype==3) THEN
            DO j = 1, n
              DO i = 1, MIN(j+1, m)
                a(i, j) = a(i, j)*mul
              END DO
            END DO
          ELSE IF (itype==4) THEN
            k3 = kl + 1
            k4 = n + 1
            DO j = 1, n
              DO i = 1, MIN(k3, k4-j)
                a(i, j) = a(i, j)*mul
              END DO
            END DO
          ELSE IF (itype==5) THEN
            k1 = ku + 2
            k3 = ku + 1
            DO j = 1, n
              DO i = MAX(k1-j, 1_ip_), k3
                a(i, j) = a(i, j)*mul
              END DO
            END DO
          ELSE IF (itype==6) THEN
            k1 = kl + ku + 2
            k2 = kl + 1
            k3 = 2*kl + ku + 1
            k4 = kl + ku + 1 + m
            DO j = 1, n
              DO i = MAX(k1-j, k2), MIN(k3, k4-j)
                a(i, j) = a(i, j)*mul
              END DO
            END DO
          END IF
          IF (.NOT. done) GO TO 10
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASD4(n, i, d, z, delta, rho, sigma, work, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: i, info, n
          REAL(rp_) :: rho, sigma
          REAL(rp_) :: d(*), delta(*), work(*), z(*)
          INTEGER(ip_) :: maxit
          PARAMETER (maxit=400)
          REAL(rp_) :: zero, one, two, three, four, eight, ten
          PARAMETER (zero=0.0_rp_, one=1.0_rp_, two=2.0_rp_, three=3.0_rp_, &
            four=4.0_rp_, eight=8.0_rp_, ten=10.0_rp_)
          LOGICAL :: orgati, swtch, swtch3, geomavg
          INTEGER(ip_) :: ii, iim1, iip1, ip1, iter, j, niter
          REAL(rp_) :: a, b, c, delsq, delsq2, sq2, dphi, dpsi, dtiim,      &
            dtiip, dtipsq, dtisq, dtnsq, dtnsq1, dw, eps, erretm, eta, phi, &
            prew, psi, rhoinv, sglb, sgub, tau, tau2, temp, temp1, temp2, w
          REAL(rp_) :: dd(3), zz(3)
          EXTERNAL :: DLAED6, DLASD5
          REAL(rp_) :: DLAMCH
          EXTERNAL :: DLAMCH
          INTRINSIC :: ABS, MAX, MIN, SQRT
          info = 0
          IF (n==1) THEN
            sigma = SQRT(d(1)*d(1)+rho*z(1)*z(1))
            delta(1) = one
            work(1) = one
            RETURN
          END IF
          IF (n==2) THEN
            CALL DLASD5(i, d, z, delta, rho, sigma, work)
            RETURN
          END IF
          eps = DLAMCH('Epsilon')
          rhoinv = one/rho
          tau2 = zero
          IF (i==n) THEN
            ii = n - 1
            niter = 1
            temp = rho/two
            temp1 = temp/(d(n)+SQRT(d(n)*d(n)+temp))
            DO j = 1, n
              work(j) = d(j) + d(n) + temp1
              delta(j) = (d(j)-d(n)) - temp1
            END DO
            psi = zero
            DO j = 1, n - 2
              psi = psi + z(j)*z(j)/(delta(j)*work(j))
            END DO
            c = rhoinv + psi
            w = c + z(ii)*z(ii)/(delta(ii)*work(ii)) +                      &
              z(n)*z(n)/(delta(n)*work(n))
            IF (w<=zero) THEN
              temp1 = SQRT(d(n)*d(n)+rho)
              temp = z(n-1)*z(n-1)/((d(n-1)+temp1)*(d(n)-d(n-1)+rho/(d(n)+  &
                temp1))) + z(n)*z(n)/rho
              IF (c<=temp) THEN
                tau = rho
              ELSE
                delsq = (d(n)-d(n-1))*(d(n)+d(n-1))
                a = -c*delsq + z(n-1)*z(n-1) + z(n)*z(n)
                b = z(n)*z(n)*delsq
                IF (a<zero) THEN
                  tau2 = two*b/(SQRT(a*a+four*b*c)-a)
                ELSE
                  tau2 = (a+SQRT(a*a+four*b*c))/(two*c)
                END IF
                tau = tau2/(d(n)+SQRT(d(n)*d(n)+tau2))
              END IF
            ELSE
              delsq = (d(n)-d(n-1))*(d(n)+d(n-1))
              a = -c*delsq + z(n-1)*z(n-1) + z(n)*z(n)
              b = z(n)*z(n)*delsq
              IF (a<zero) THEN
                tau2 = two*b/(SQRT(a*a+four*b*c)-a)
              ELSE
                tau2 = (a+SQRT(a*a+four*b*c))/(two*c)
              END IF
              tau = tau2/(d(n)+SQRT(d(n)*d(n)+tau2))
            END IF
            sigma = d(n) + tau
            DO j = 1, n
              delta(j) = (d(j)-d(n)) - tau
              work(j) = d(j) + d(n) + tau
            END DO
            dpsi = zero
            psi = zero
            erretm = zero
            DO j = 1, ii
              temp = z(j)/(delta(j)*work(j))
              psi = psi + z(j)*temp
              dpsi = dpsi + temp*temp
              erretm = erretm + psi
            END DO
            erretm = ABS(erretm)
            temp = z(n)/(delta(n)*work(n))
            phi = z(n)*temp
            dphi = temp*temp
            erretm = eight*(-phi-psi) + erretm - phi + rhoinv
            w = rhoinv + phi + psi
            IF (ABS(w)<=eps*erretm) THEN
              GO TO 240
            END IF
            niter = niter + 1
            dtnsq1 = work(n-1)*delta(n-1)
            dtnsq = work(n)*delta(n)
            c = w - dtnsq1*dpsi - dtnsq*dphi
            a = (dtnsq+dtnsq1)*w - dtnsq*dtnsq1*(dpsi+dphi)
            b = dtnsq*dtnsq1*w
            IF (c<zero) c = ABS(c)
            IF (c==zero) THEN
              eta = rho - sigma*sigma
            ELSE IF (a>=zero) THEN
              eta = (a+SQRT(ABS(a*a-four*b*c)))/(two*c)
            ELSE
              eta = two*b/(a-SQRT(ABS(a*a-four*b*c)))
            END IF
            IF (w*eta>zero) eta = -w/(dpsi+dphi)
            temp = eta - dtnsq
            IF (temp>rho) eta = rho + dtnsq
            eta = eta/(sigma+SQRT(eta+sigma*sigma))
            tau = tau + eta
            sigma = sigma + eta
            DO j = 1, n
              delta(j) = delta(j) - eta
              work(j) = work(j) + eta
            END DO
            dpsi = zero
            psi = zero
            erretm = zero
            DO j = 1, ii
              temp = z(j)/(work(j)*delta(j))
              psi = psi + z(j)*temp
              dpsi = dpsi + temp*temp
              erretm = erretm + psi
            END DO
            erretm = ABS(erretm)
            tau2 = work(n)*delta(n)
            temp = z(n)/tau2
            phi = z(n)*temp
            dphi = temp*temp
            erretm = eight*(-phi-psi) + erretm - phi + rhoinv
            w = rhoinv + phi + psi
            iter = niter + 1
            DO niter = iter, maxit
              IF (ABS(w)<=eps*erretm) THEN
                GO TO 240
              END IF
              dtnsq1 = work(n-1)*delta(n-1)
              dtnsq = work(n)*delta(n)
              c = w - dtnsq1*dpsi - dtnsq*dphi
              a = (dtnsq+dtnsq1)*w - dtnsq1*dtnsq*(dpsi+dphi)
              b = dtnsq1*dtnsq*w
              IF (a>=zero) THEN
                eta = (a+SQRT(ABS(a*a-four*b*c)))/(two*c)
              ELSE
                eta = two*b/(a-SQRT(ABS(a*a-four*b*c)))
              END IF
              IF (w*eta>zero) eta = -w/(dpsi+dphi)
              temp = eta - dtnsq
              IF (temp<=zero) eta = eta/two
              eta = eta/(sigma+SQRT(eta+sigma*sigma))
              tau = tau + eta
              sigma = sigma + eta
              DO j = 1, n
                delta(j) = delta(j) - eta
                work(j) = work(j) + eta
              END DO
              dpsi = zero
              psi = zero
              erretm = zero
              DO j = 1, ii
                temp = z(j)/(work(j)*delta(j))
                psi = psi + z(j)*temp
                dpsi = dpsi + temp*temp
                erretm = erretm + psi
              END DO
              erretm = ABS(erretm)
              tau2 = work(n)*delta(n)
              temp = z(n)/tau2
              phi = z(n)*temp
              dphi = temp*temp
              erretm = eight*(-phi-psi) + erretm - phi + rhoinv
              w = rhoinv + phi + psi
            END DO
            info = 1
            GO TO 240
          ELSE
            niter = 1
            ip1 = i + 1
            delsq = (d(ip1)-d(i))*(d(ip1)+d(i))
            delsq2 = delsq/two
            sq2 = SQRT((d(i)*d(i)+d(ip1)*d(ip1))/two)
            temp = delsq2/(d(i)+sq2)
            DO j = 1, n
              work(j) = d(j) + d(i) + temp
              delta(j) = (d(j)-d(i)) - temp
            END DO
            psi = zero
            DO j = 1, i - 1
              psi = psi + z(j)*z(j)/(work(j)*delta(j))
            END DO
            phi = zero
            DO j = n, i + 2, -1_ip_
              phi = phi + z(j)*z(j)/(work(j)*delta(j))
            END DO
            c = rhoinv + psi + phi
            w = c + z(i)*z(i)/(work(i)*delta(i)) +                          &
              z(ip1)*z(ip1)/(work(ip1)*delta(ip1))
            geomavg = .FALSE.
            IF (w>zero) THEN
              orgati = .TRUE.
              ii = i
              sglb = zero
              sgub = delsq2/(d(i)+sq2)
              a = c*delsq + z(i)*z(i) + z(ip1)*z(ip1)
              b = z(i)*z(i)*delsq
              IF (a>zero) THEN
                tau2 = two*b/(a+SQRT(ABS(a*a-four*b*c)))
              ELSE
                tau2 = (a-SQRT(ABS(a*a-four*b*c)))/(two*c)
              END IF
              tau = tau2/(d(i)+SQRT(d(i)*d(i)+tau2))
              temp = SQRT(eps)
              IF ((d(i)<=temp*d(ip1)) .AND. (ABS(z(i))<=temp) .AND. (d(     &
                i)>zero)) THEN
                tau = MIN(ten*d(i), sgub)
                geomavg = .TRUE.
              END IF
            ELSE
              orgati = .FALSE.
              ii = ip1
              sglb = -delsq2/(d(ii)+sq2)
              sgub = zero
              a = c*delsq - z(i)*z(i) - z(ip1)*z(ip1)
              b = z(ip1)*z(ip1)*delsq
              IF (a<zero) THEN
                tau2 = two*b/(a-SQRT(ABS(a*a+four*b*c)))
              ELSE
                tau2 = -(a+SQRT(ABS(a*a+four*b*c)))/(two*c)
              END IF
              tau = tau2/(d(ip1)+SQRT(ABS(d(ip1)*d(ip1)+tau2)))
            END IF
            sigma = d(ii) + tau
            DO j = 1, n
              work(j) = d(j) + d(ii) + tau
              delta(j) = (d(j)-d(ii)) - tau
            END DO
            iim1 = ii - 1
            iip1 = ii + 1
            dpsi = zero
            psi = zero
            erretm = zero
            DO j = 1, iim1
              temp = z(j)/(work(j)*delta(j))
              psi = psi + z(j)*temp
              dpsi = dpsi + temp*temp
              erretm = erretm + psi
            END DO
            erretm = ABS(erretm)
            dphi = zero
            phi = zero
            DO j = n, iip1, -1_ip_
              temp = z(j)/(work(j)*delta(j))
              phi = phi + z(j)*temp
              dphi = dphi + temp*temp
              erretm = erretm + phi
            END DO
            w = rhoinv + phi + psi
            swtch3 = .FALSE.
            IF (orgati) THEN
              IF (w<zero) swtch3 = .TRUE.
            ELSE
              IF (w>zero) swtch3 = .TRUE.
            END IF
            IF (ii==1 .OR. ii==n) swtch3 = .FALSE.
            temp = z(ii)/(work(ii)*delta(ii))
            dw = dpsi + dphi + temp*temp
            temp = z(ii)*temp
            w = w + temp
            erretm = eight*(phi-psi) + erretm + two*rhoinv +                &
              three*ABS(temp)
            IF (ABS(w)<=eps*erretm) THEN
              GO TO 240
            END IF
            IF (w<=zero) THEN
              sglb = MAX(sglb, tau)
            ELSE
              sgub = MIN(sgub, tau)
            END IF
            niter = niter + 1
            IF (.NOT. swtch3) THEN
              dtipsq = work(ip1)*delta(ip1)
              dtisq = work(i)*delta(i)
              IF (orgati) THEN
                c = w - dtipsq*dw + delsq*(z(i)/dtisq)**2
              ELSE
                c = w - dtisq*dw - delsq*(z(ip1)/dtipsq)**2
              END IF
              a = (dtipsq+dtisq)*w - dtipsq*dtisq*dw
              b = dtipsq*dtisq*w
              IF (c==zero) THEN
                IF (a==zero) THEN
                  IF (orgati) THEN
                    a = z(i)*z(i) + dtipsq*dtipsq*(dpsi+dphi)
                  ELSE
                    a = z(ip1)*z(ip1) + dtisq*dtisq*(dpsi+dphi)
                  END IF
                END IF
                eta = b/a
              ELSE IF (a<=zero) THEN
                eta = (a-SQRT(ABS(a*a-four*b*c)))/(two*c)
              ELSE
                eta = two*b/(a+SQRT(ABS(a*a-four*b*c)))
              END IF
            ELSE
              dtiim = work(iim1)*delta(iim1)
              dtiip = work(iip1)*delta(iip1)
              temp = rhoinv + psi + phi
              IF (orgati) THEN
                temp1 = z(iim1)/dtiim
                temp1 = temp1*temp1
                c = (temp-dtiip*(dpsi+dphi)) - (d(iim1)-d(iip1))*(d(iim1)+  &
                  d(iip1))*temp1
                zz(1) = z(iim1)*z(iim1)
                IF (dpsi<temp1) THEN
                  zz(3) = dtiip*dtiip*dphi
                ELSE
                  zz(3) = dtiip*dtiip*((dpsi-temp1)+dphi)
                END IF
              ELSE
                temp1 = z(iip1)/dtiip
                temp1 = temp1*temp1
                c = (temp-dtiim*(dpsi+dphi)) - (d(iip1)-d(iim1))*(d(iim1)+  &
                  d(iip1))*temp1
                IF (dphi<temp1) THEN
                  zz(1) = dtiim*dtiim*dpsi
                ELSE
                  zz(1) = dtiim*dtiim*(dpsi+(dphi-temp1))
                END IF
                zz(3) = z(iip1)*z(iip1)
              END IF
              zz(2) = z(ii)*z(ii)
              dd(1) = dtiim
              dd(2) = delta(ii)*work(ii)
              dd(3) = dtiip
              CALL DLAED6(niter, orgati, c, dd, zz, w, eta, info)
              IF (info/=0) THEN
                swtch3 = .FALSE.
                info = 0
                dtipsq = work(ip1)*delta(ip1)
                dtisq = work(i)*delta(i)
                IF (orgati) THEN
                  c = w - dtipsq*dw + delsq*(z(i)/dtisq)**2
                ELSE
                  c = w - dtisq*dw - delsq*(z(ip1)/dtipsq)**2
                END IF
                a = (dtipsq+dtisq)*w - dtipsq*dtisq*dw
                b = dtipsq*dtisq*w
                IF (c==zero) THEN
                  IF (a==zero) THEN
                    IF (orgati) THEN
                      a = z(i)*z(i) + dtipsq*dtipsq*(dpsi+dphi)
                    ELSE
                      a = z(ip1)*z(ip1) + dtisq*dtisq*(dpsi+dphi)
                    END IF
                  END IF
                  eta = b/a
                ELSE IF (a<=zero) THEN
                  eta = (a-SQRT(ABS(a*a-four*b*c)))/(two*c)
                ELSE
                  eta = two*b/(a+SQRT(ABS(a*a-four*b*c)))
                END IF
              END IF
            END IF
            IF (w*eta>=zero) eta = -w/dw
            eta = eta/(sigma+SQRT(sigma*sigma+eta))
            temp = tau + eta
            IF (temp>sgub .OR. temp<sglb) THEN
              IF (w<zero) THEN
                eta = (sgub-tau)/two
              ELSE
                eta = (sglb-tau)/two
              END IF
              IF (geomavg) THEN
                IF (w<zero) THEN
                  IF (tau>zero) THEN
                    eta = SQRT(sgub*tau) - tau
                  END IF
                ELSE
                  IF (sglb>zero) THEN
                    eta = SQRT(sglb*tau) - tau
                  END IF
                END IF
              END IF
            END IF
            prew = w
            tau = tau + eta
            sigma = sigma + eta
            DO j = 1, n
              work(j) = work(j) + eta
              delta(j) = delta(j) - eta
            END DO
            dpsi = zero
            psi = zero
            erretm = zero
            DO j = 1, iim1
              temp = z(j)/(work(j)*delta(j))
              psi = psi + z(j)*temp
              dpsi = dpsi + temp*temp
              erretm = erretm + psi
            END DO
            erretm = ABS(erretm)
            dphi = zero
            phi = zero
            DO j = n, iip1, -1_ip_
              temp = z(j)/(work(j)*delta(j))
              phi = phi + z(j)*temp
              dphi = dphi + temp*temp
              erretm = erretm + phi
            END DO
            tau2 = work(ii)*delta(ii)
            temp = z(ii)/tau2
            dw = dpsi + dphi + temp*temp
            temp = z(ii)*temp
            w = rhoinv + phi + psi + temp
            erretm = eight*(phi-psi) + erretm + two*rhoinv +                &
              three*ABS(temp)
            swtch = .FALSE.
            IF (orgati) THEN
              IF (-w>ABS(prew)/ten) swtch = .TRUE.
            ELSE
              IF (w>ABS(prew)/ten) swtch = .TRUE.
            END IF
            iter = niter + 1
            DO niter = iter, maxit
              IF (ABS(w)<=eps*erretm) THEN
                GO TO 240
              END IF
              IF (w<=zero) THEN
                sglb = MAX(sglb, tau)
              ELSE
                sgub = MIN(sgub, tau)
              END IF
              IF (.NOT. swtch3) THEN
                dtipsq = work(ip1)*delta(ip1)
                dtisq = work(i)*delta(i)
                IF (.NOT. swtch) THEN
                  IF (orgati) THEN
                    c = w - dtipsq*dw + delsq*(z(i)/dtisq)**2
                  ELSE
                    c = w - dtisq*dw - delsq*(z(ip1)/dtipsq)**2
                  END IF
                ELSE
                  temp = z(ii)/(work(ii)*delta(ii))
                  IF (orgati) THEN
                    dpsi = dpsi + temp*temp
                  ELSE
                    dphi = dphi + temp*temp
                  END IF
                  c = w - dtisq*dpsi - dtipsq*dphi
                END IF
                a = (dtipsq+dtisq)*w - dtipsq*dtisq*dw
                b = dtipsq*dtisq*w
                IF (c==zero) THEN
                  IF (a==zero) THEN
                    IF (.NOT. swtch) THEN
                      IF (orgati) THEN
                        a = z(i)*z(i) + dtipsq*dtipsq*(dpsi+dphi)
                      ELSE
                        a = z(ip1)*z(ip1) + dtisq*dtisq*(dpsi+dphi)
                      END IF
                    ELSE
                      a = dtisq*dtisq*dpsi + dtipsq*dtipsq*dphi
                    END IF
                  END IF
                  eta = b/a
                ELSE IF (a<=zero) THEN
                  eta = (a-SQRT(ABS(a*a-four*b*c)))/(two*c)
                ELSE
                  eta = two*b/(a+SQRT(ABS(a*a-four*b*c)))
                END IF
              ELSE
                dtiim = work(iim1)*delta(iim1)
                dtiip = work(iip1)*delta(iip1)
                temp = rhoinv + psi + phi
                IF (swtch) THEN
                  c = temp - dtiim*dpsi - dtiip*dphi
                  zz(1) = dtiim*dtiim*dpsi
                  zz(3) = dtiip*dtiip*dphi
                ELSE
                  IF (orgati) THEN
                    temp1 = z(iim1)/dtiim
                    temp1 = temp1*temp1
                    temp2 = (d(iim1)-d(iip1))*(d(iim1)+d(iip1))*temp1
                    c = temp - dtiip*(dpsi+dphi) - temp2
                    zz(1) = z(iim1)*z(iim1)
                    IF (dpsi<temp1) THEN
                      zz(3) = dtiip*dtiip*dphi
                    ELSE
                      zz(3) = dtiip*dtiip*((dpsi-temp1)+dphi)
                    END IF
                  ELSE
                    temp1 = z(iip1)/dtiip
                    temp1 = temp1*temp1
                    temp2 = (d(iip1)-d(iim1))*(d(iim1)+d(iip1))*temp1
                    c = temp - dtiim*(dpsi+dphi) - temp2
                    IF (dphi<temp1) THEN
                      zz(1) = dtiim*dtiim*dpsi
                    ELSE
                      zz(1) = dtiim*dtiim*(dpsi+(dphi-temp1))
                    END IF
                    zz(3) = z(iip1)*z(iip1)
                  END IF
                END IF
                dd(1) = dtiim
                dd(2) = delta(ii)*work(ii)
                dd(3) = dtiip
                CALL DLAED6(niter, orgati, c, dd, zz, w, eta, info)
                IF (info/=0) THEN
                  swtch3 = .FALSE.
                  info = 0
                  dtipsq = work(ip1)*delta(ip1)
                  dtisq = work(i)*delta(i)
                  IF (.NOT. swtch) THEN
                    IF (orgati) THEN
                      c = w - dtipsq*dw + delsq*(z(i)/dtisq)**2
                    ELSE
                      c = w - dtisq*dw - delsq*(z(ip1)/dtipsq)**2
                    END IF
                  ELSE
                    temp = z(ii)/(work(ii)*delta(ii))
                    IF (orgati) THEN
                      dpsi = dpsi + temp*temp
                    ELSE
                      dphi = dphi + temp*temp
                    END IF
                    c = w - dtisq*dpsi - dtipsq*dphi
                  END IF
                  a = (dtipsq+dtisq)*w - dtipsq*dtisq*dw
                  b = dtipsq*dtisq*w
                  IF (c==zero) THEN
                    IF (a==zero) THEN
                      IF (.NOT. swtch) THEN
                        IF (orgati) THEN
                          a = z(i)*z(i) + dtipsq*dtipsq*(dpsi+dphi)
                        ELSE
                          a = z(ip1)*z(ip1) + dtisq*dtisq*(dpsi+dphi)
                        END IF
                      ELSE
                        a = dtisq*dtisq*dpsi + dtipsq*dtipsq*dphi
                      END IF
                    END IF
                    eta = b/a
                  ELSE IF (a<=zero) THEN
                    eta = (a-SQRT(ABS(a*a-four*b*c)))/(two*c)
                  ELSE
                    eta = two*b/(a+SQRT(ABS(a*a-four*b*c)))
                  END IF
                END IF
              END IF
              IF (w*eta>=zero) eta = -w/dw
              eta = eta/(sigma+SQRT(sigma*sigma+eta))
              temp = tau + eta
              IF (temp>sgub .OR. temp<sglb) THEN
                IF (w<zero) THEN
                  eta = (sgub-tau)/two
                ELSE
                  eta = (sglb-tau)/two
                END IF
                IF (geomavg) THEN
                  IF (w<zero) THEN
                    IF (tau>zero) THEN
                      eta = SQRT(sgub*tau) - tau
                    END IF
                  ELSE
                    IF (sglb>zero) THEN
                      eta = SQRT(sglb*tau) - tau
                    END IF
                  END IF
                END IF
              END IF
              prew = w
              tau = tau + eta
              sigma = sigma + eta
              DO j = 1, n
                work(j) = work(j) + eta
                delta(j) = delta(j) - eta
              END DO
              dpsi = zero
              psi = zero
              erretm = zero
              DO j = 1, iim1
                temp = z(j)/(work(j)*delta(j))
                psi = psi + z(j)*temp
                dpsi = dpsi + temp*temp
                erretm = erretm + psi
              END DO
              erretm = ABS(erretm)
              dphi = zero
              phi = zero
              DO j = n, iip1, -1_ip_
                temp = z(j)/(work(j)*delta(j))
                phi = phi + z(j)*temp
                dphi = dphi + temp*temp
                erretm = erretm + phi
              END DO
              tau2 = work(ii)*delta(ii)
              temp = z(ii)/tau2
              dw = dpsi + dphi + temp*temp
              temp = z(ii)*temp
              w = rhoinv + phi + psi + temp
              erretm = eight*(phi-psi) + erretm + two*rhoinv +              &
                three*ABS(temp)
              IF (w*prew>zero .AND. ABS(w)>ABS(prew)/ten) swtch = .NOT.     &
                swtch
            END DO
            info = 1
          END IF
 240      CONTINUE
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASD5(i, d, z, delta, rho, dsigma, work)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: i
          REAL(rp_) :: dsigma, rho
          REAL(rp_) :: d(2), delta(2), work(2), z(2)
          REAL(rp_) :: zero, one, two, three, four
          PARAMETER (zero=0.0_rp_, one=1.0_rp_, two=2.0_rp_, three=3.0_rp_, &
            four=4.0_rp_)
          REAL(rp_) :: b, c, del, delsq, tau, w
          INTRINSIC :: ABS, SQRT
          del = d(2) - d(1)
          delsq = del*(d(2)+d(1))
          IF (i==1) THEN
            w = one + four*rho*(z(2)*z(2)/(d(1)+three*d(2))-z(1)*z(1)/(     &
              three*d(1)+d(2)))/del
            IF (w>zero) THEN
              b = delsq + rho*(z(1)*z(1)+z(2)*z(2))
              c = rho*z(1)*z(1)*delsq
              tau = two*c/(b+SQRT(ABS(b*b-four*c)))
              tau = tau/(d(1)+SQRT(d(1)*d(1)+tau))
              dsigma = d(1) + tau
              delta(1) = -tau
              delta(2) = del - tau
              work(1) = two*d(1) + tau
              work(2) = (d(1)+tau) + d(2)
            ELSE
              b = -delsq + rho*(z(1)*z(1)+z(2)*z(2))
              c = rho*z(2)*z(2)*delsq
              IF (b>zero) THEN
                tau = -two*c/(b+SQRT(b*b+four*c))
              ELSE
                tau = (b-SQRT(b*b+four*c))/two
              END IF
              tau = tau/(d(2)+SQRT(ABS(d(2)*d(2)+tau)))
              dsigma = d(2) + tau
              delta(1) = -(del+tau)
              delta(2) = -tau
              work(1) = d(1) + tau + d(2)
              work(2) = two*d(2) + tau
            END IF
          ELSE
            b = -delsq + rho*(z(1)*z(1)+z(2)*z(2))
            c = rho*z(2)*z(2)*delsq
            IF (b>zero) THEN
              tau = (b+SQRT(b*b+four*c))/two
            ELSE
              tau = two*c/(-b+SQRT(b*b+four*c))
            END IF
            tau = tau/(d(2)+SQRT(d(2)*d(2)+tau))
            dsigma = d(2) + tau
            delta(1) = -(del+tau)
            delta(2) = -tau
            work(1) = d(1) + tau + d(2)
            work(2) = two*d(2) + tau
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASD6(icompq, nl, nr, sqre, d, vf, vl, alpha, beta,     &
          idxq, perm, givptr, givcol, ldgcol, givnum, ldgnum, poles, difl,  &
          difr, z, k, c, s, work, iwork, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: givptr, icompq, info, k, ldgcol, ldgnum, nl, nr,  &
            sqre
          REAL(rp_) :: alpha, beta, c, s
          INTEGER(ip_) :: givcol(ldgcol, *), idxq(*), iwork(*), perm(*)
          REAL(rp_) :: d(*), difl(*), difr(*), givnum(ldgnum, *),           &
            poles(ldgnum, *), vf(*), vl(*), work(*), z(*)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          INTEGER(ip_) :: i, idx, idxc, idxp, isigma, ivfw, ivlw, iw, m, n, &
            n1, n2
          REAL(rp_) :: orgnrm
          EXTERNAL :: DCOPY, DLAMRG, DLASCL, DLASD7, &
            DLASD8, XERBLA2
          INTRINSIC :: ABS, MAX
          info = 0
          n = nl + nr + 1
          m = n + sqre
          IF ((icompq<0) .OR. (icompq>1)) THEN
            info = -1
          ELSE IF (nl<1) THEN
            info = -2
          ELSE IF (nr<1) THEN
            info = -3
          ELSE IF ((sqre<0) .OR. (sqre>1)) THEN
            info = -4
          ELSE IF (ldgcol<n) THEN
            info = -14
          ELSE IF (ldgnum<n) THEN
            info = -16
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('LASD6', -info)
            RETURN
          END IF
          isigma = 1
          iw = isigma + n
          ivfw = iw + m
          ivlw = ivfw + m
          idx = 1
          idxc = idx + n
          idxp = idxc + n
          orgnrm = MAX(ABS(alpha), ABS(beta))
          d(nl+1) = zero
          DO i = 1, n
            IF (ABS(d(i))>orgnrm) THEN
              orgnrm = ABS(d(i))
            END IF
          END DO
          CALL DLASCL('G', 0_ip_, 0_ip_, orgnrm, one, n, 1_ip_, d, n, info)
          alpha = alpha/orgnrm
          beta = beta/orgnrm
          CALL DLASD7(icompq, nl, nr, sqre, k, d, z, work(iw), vf,          &
            work(ivfw), vl, work(ivlw), alpha, beta, work(isigma),          &
            iwork(idx), iwork(idxp), idxq, perm, givptr, givcol, ldgcol,    &
            givnum, ldgnum, c, s, info)
          CALL DLASD8(icompq, k, d, z, vf, vl, difl, difr, ldgnum,          &
            work(isigma), work(iw), info)
          IF (info/=0) THEN
            RETURN
          END IF
          IF (icompq==1) THEN
            CALL DCOPY(k, d, 1_ip_, poles(1,1), 1_ip_)
            CALL DCOPY(k, work(isigma), 1_ip_, poles(1,2), 1_ip_)
          END IF
          CALL DLASCL('G', 0_ip_, 0_ip_, one, orgnrm, n, 1_ip_, d, n, info)
          n1 = k
          n2 = n - k
          CALL DLAMRG(n1, n2, d, 1_ip_, -1_ip_, idxq)
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASD7(icompq, nl, nr, sqre, k, d, z, zw, vf, vfw, vl,   &
          vlw, alpha, beta, dsigma, idx, idxp, idxq, perm, givptr, givcol,  &
          ldgcol, givnum, ldgnum, c, s, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: givptr, icompq, info, k, ldgcol, ldgnum, nl, nr,  &
            sqre
          REAL(rp_) :: alpha, beta, c, s
          INTEGER(ip_) :: givcol(ldgcol, *), idx(*), idxp(*), idxq(*),      &
            perm(*)
          REAL(rp_) :: d(*), dsigma(*), givnum(ldgnum, *), vf(*), vfw(*),   &
            vl(*), vlw(*), z(*), zw(*)
          REAL(rp_) :: zero, one, two, eight
          PARAMETER (zero=0.0_rp_, one=1.0_rp_, two=2.0_rp_, eight=8.0_rp_)
          INTEGER(ip_) :: i, idxi, idxj, idxjp, j, jp, jprev, k2, m, n,     &
            nlp1, nlp2
          REAL(rp_) :: eps, hlftol, tau, tol, z1
          EXTERNAL :: DCOPY, DLAMRG, DROT, XERBLA2
          REAL(rp_) :: DLAMCH, DLAPY2
          EXTERNAL :: DLAMCH, DLAPY2
          INTRINSIC :: ABS, MAX
          info = 0
          n = nl + nr + 1
          m = n + sqre
          IF ((icompq<0) .OR. (icompq>1)) THEN
            info = -1
          ELSE IF (nl<1) THEN
            info = -2
          ELSE IF (nr<1) THEN
            info = -3
          ELSE IF ((sqre<0) .OR. (sqre>1)) THEN
            info = -4
          ELSE IF (ldgcol<n) THEN
            info = -22
          ELSE IF (ldgnum<n) THEN
            info = -24
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('LASD7', -info)
            RETURN
          END IF
          nlp1 = nl + 1
          nlp2 = nl + 2
          IF (icompq==1) THEN
            givptr = 0
          END IF
          z1 = alpha*vl(nlp1)
          vl(nlp1) = zero
          tau = vf(nlp1)
          DO i = nl, 1_ip_, -1_ip_
            z(i+1) = alpha*vl(i)
            vl(i) = zero
            vf(i+1) = vf(i)
            d(i+1) = d(i)
            idxq(i+1) = idxq(i) + 1
          END DO
          vf(1) = tau
          DO i = nlp2, m
            z(i) = beta*vf(i)
            vf(i) = zero
          END DO
          DO i = nlp2, n
            idxq(i) = idxq(i) + nlp1
          END DO
          DO i = 2, n
            dsigma(i) = d(idxq(i))
            zw(i) = z(idxq(i))
            vfw(i) = vf(idxq(i))
            vlw(i) = vl(idxq(i))
          END DO
          CALL DLAMRG(nl, nr, dsigma(2), 1_ip_, 1_ip_, idx(2))
          DO i = 2, n
            idxi = 1 + idx(i)
            d(i) = dsigma(idxi)
            z(i) = zw(idxi)
            vf(i) = vfw(idxi)
            vl(i) = vlw(idxi)
          END DO
          eps = DLAMCH('Epsilon')
          tol = MAX(ABS(alpha), ABS(beta))
          tol = eight*eight*eps*MAX(ABS(d(n)), tol)
          k = 1
          k2 = n + 1
          DO j = 2, n
            IF (ABS(z(j))<=tol) THEN
              k2 = k2 - 1
              idxp(k2) = j
              IF (j==n) GO TO 100
            ELSE
              jprev = j
              GO TO 70
            END IF
          END DO
 70       CONTINUE
          j = jprev
 80       CONTINUE
          j = j + 1
          IF (j>n) GO TO 90
          IF (ABS(z(j))<=tol) THEN
            k2 = k2 - 1
            idxp(k2) = j
          ELSE
            IF (ABS(d(j)-d(jprev))<=tol) THEN
              s = z(jprev)
              c = z(j)
              tau = DLAPY2(c, s)
              z(j) = tau
              z(jprev) = zero
              c = c/tau
              s = -s/tau
              IF (icompq==1) THEN
                givptr = givptr + 1
                idxjp = idxq(idx(jprev)+1)
                idxj = idxq(idx(j)+1)
                IF (idxjp<=nlp1) THEN
                  idxjp = idxjp - 1
                END IF
                IF (idxj<=nlp1) THEN
                  idxj = idxj - 1
                END IF
                givcol(givptr, 2_ip_) = idxjp
                givcol(givptr, 1_ip_) = idxj
                givnum(givptr, 2_ip_) = c
                givnum(givptr, 1_ip_) = s
              END IF
              CALL DROT(1_ip_, vf(jprev), 1_ip_, vf(j), 1_ip_, c, s)
              CALL DROT(1_ip_, vl(jprev), 1_ip_, vl(j), 1_ip_, c, s)
              k2 = k2 - 1
              idxp(k2) = jprev
              jprev = j
            ELSE
              k = k + 1
              zw(k) = z(jprev)
              dsigma(k) = d(jprev)
              idxp(k) = jprev
              jprev = j
            END IF
          END IF
          GO TO 80
 90       CONTINUE
          k = k + 1
          zw(k) = z(jprev)
          dsigma(k) = d(jprev)
          idxp(k) = jprev
 100      CONTINUE
          DO j = 2, n
            jp = idxp(j)
            dsigma(j) = d(jp)
            vfw(j) = vf(jp)
            vlw(j) = vl(jp)
          END DO
          IF (icompq==1) THEN
            DO j = 2, n
              jp = idxp(j)
              perm(j) = idxq(idx(jp)+1)
              IF (perm(j)<=nlp1) THEN
                perm(j) = perm(j) - 1
              END IF
            END DO
          END IF
          CALL DCOPY(n-k, dsigma(k+1), 1_ip_, d(k+1), 1_ip_)
          dsigma(1) = zero
          hlftol = tol/two
          IF (ABS(dsigma(2))<=hlftol) dsigma(2) = hlftol
          IF (m>n) THEN
            z(1) = DLAPY2(z1, z(m))
            IF (z(1)<=tol) THEN
              c = one
              s = zero
              z(1) = tol
            ELSE
              c = z1/z(1)
              s = -z(m)/z(1)
            END IF
            CALL DROT(1_ip_, vf(m), 1_ip_, vf(1), 1_ip_, c, s)
            CALL DROT(1_ip_, vl(m), 1_ip_, vl(1), 1_ip_, c, s)
          ELSE
            IF (ABS(z1)<=tol) THEN
              z(1) = tol
            ELSE
              z(1) = z1
            END IF
          END IF
          CALL DCOPY(k-1, zw(2), 1_ip_, z(2), 1_ip_)
          CALL DCOPY(n-1, vfw(2), 1_ip_, vf(2), 1_ip_)
          CALL DCOPY(n-1, vlw(2), 1_ip_, vl(2), 1_ip_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASD8(icompq, k, d, z, vf, vl, difl, difr, lddifr,      &
          dsigma, work, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: icompq, info, k, lddifr
          REAL(rp_) :: d(*), difl(*), difr(lddifr, *), dsigma(*), vf(*),    &
            vl(*), work(*), z(*)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          INTEGER(ip_) :: i, iwk1, iwk2, iwk2i, iwk3, iwk3i, j
          REAL(rp_) :: diflj, difrj, dj, dsigj, dsigjp, rho, temp
          EXTERNAL :: DCOPY, DLASCL, DLASD4, DLASET, XERBLA2
          REAL(rp_) :: DDOT, DLAMC3, DNRM2
          EXTERNAL :: DDOT, DLAMC3, DNRM2
          INTRINSIC :: ABS, SIGN, SQRT
          info = 0
          IF ((icompq<0) .OR. (icompq>1)) THEN
            info = -1
          ELSE IF (k<1) THEN
            info = -2
          ELSE IF (lddifr<k) THEN
            info = -9
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('LASD8', -info)
            RETURN
          END IF
          IF (k==1) THEN
            d(1) = ABS(z(1))
            difl(1) = d(1)
            IF (icompq==1) THEN
              difl(2) = one
              difr(1, 2_ip_) = one
            END IF
            RETURN
          END IF
          DO i = 1, k
            dsigma(i) = DLAMC3(dsigma(i), dsigma(i)) - dsigma(i)
          END DO
          iwk1 = 1
          iwk2 = iwk1 + k
          iwk3 = iwk2 + k
          iwk2i = iwk2 - 1
          iwk3i = iwk3 - 1
          rho = DNRM2(k, z, 1_ip_)
          CALL DLASCL('G', 0_ip_, 0_ip_, rho, one, k, 1_ip_, z, k, info)
          rho = rho*rho
          CALL DLASET('A', k, 1_ip_, one, one, work(iwk3), k)
          DO j = 1, k
            CALL DLASD4(k, j, dsigma, z, work(iwk1), rho, d(j), work(iwk2), &
              info)
            IF (info/=0) THEN
              RETURN
            END IF
            work(iwk3i+j) = work(iwk3i+j)*work(j)*work(iwk2i+j)
            difl(j) = -work(j)
            difr(j, 1_ip_) = -work(j+1)
            DO i = 1, j - 1
              work(iwk3i+i) = work(iwk3i+i)*work(i)*work(iwk2i+i)/          &
                (dsigma(i)-dsigma(j))/(dsigma(i)+dsigma(j))
            END DO
            DO i = j + 1, k
              work(iwk3i+i) = work(iwk3i+i)*work(i)*work(iwk2i+i)/          &
                (dsigma(i)-dsigma(j))/(dsigma(i)+dsigma(j))
            END DO
          END DO
          DO i = 1, k
            z(i) = SIGN(SQRT(ABS(work(iwk3i+i))), z(i))
          END DO
          DO j = 1, k
            diflj = difl(j)
            dj = d(j)
            dsigj = -dsigma(j)
            IF (j<k) THEN
              difrj = -difr(j, 1_ip_)
              dsigjp = -dsigma(j+1)
            END IF
            work(j) = -z(j)/diflj/(dsigma(j)+dj)
            DO i = 1, j - 1
              work(i) = z(i)/(DLAMC3(dsigma(i),dsigj)-diflj)/               &
                (dsigma(i)+dj)
            END DO
            DO i = j + 1, k
              work(i) = z(i)/(DLAMC3(dsigma(i),dsigjp)+difrj)/              &
                (dsigma(i)+dj)
            END DO
            temp = DNRM2(k, work, 1_ip_)
            work(iwk2i+j) = DDOT(k, work, 1_ip_, vf, 1_ip_)/temp
            work(iwk3i+j) = DDOT(k, work, 1_ip_, vl, 1_ip_)/temp
            IF (icompq==1) THEN
              difr(j, 2_ip_) = temp
            END IF
          END DO
          CALL DCOPY(k, work(iwk2), 1_ip_, vf, 1_ip_)
          CALL DCOPY(k, work(iwk3), 1_ip_, vl, 1_ip_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASDA(icompq, smlsiz, n, sqre, d, e, u, ldu, vt, k,     &
          difl, difr, z, poles, givptr, givcol, ldgcol, perm, givnum, c, s, &
          work, iwork, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: icompq, info, ldgcol, ldu, n, smlsiz, sqre
          INTEGER(ip_) :: givcol(ldgcol, *), givptr(*), iwork(*), k(*),     &
            perm(ldgcol, *)
          REAL(rp_) :: c(*), d(*), difl(ldu, *), difr(ldu, *), e(*),        &
            givnum(ldu, *), poles(ldu, *), s(*), u(ldu, *), vt(ldu, *),     &
            work(*), z(ldu, *)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          INTEGER(ip_) :: i, i1, ic, idxq, idxqi, im1, inode, itemp, iwk,   &
            j, lf, ll, lvl, lvl2, m, ncc, nd, ndb1, ndiml, ndimr, nl, nlf,  &
            nlp1, nlvl, nr, nrf, nrp1, nru, nwork1, nwork2, smlszp, sqrei,  &
            vf, vfi, vl, vli
          REAL(rp_) :: alpha, beta
          EXTERNAL :: DCOPY, DLASD6, DLASDQ, DLASDT, &
            DLASET, XERBLA2
          info = 0
          IF ((icompq<0) .OR. (icompq>1)) THEN
            info = -1
          ELSE IF (smlsiz<3) THEN
            info = -2
          ELSE IF (n<0) THEN
            info = -3
          ELSE IF ((sqre<0) .OR. (sqre>1)) THEN
            info = -4
          ELSE IF (ldu<(n+sqre)) THEN
            info = -8
          ELSE IF (ldgcol<n) THEN
            info = -17
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('LASDA', -info)
            RETURN
          END IF
          m = n + sqre
          IF (n<=smlsiz) THEN
            IF (icompq==0) THEN
              CALL DLASDQ('U', sqre, n, 0_ip_, 0_ip_, 0_ip_, d, e, vt, ldu, &
                u, ldu, u, ldu, work, info)
            ELSE
              CALL DLASDQ('U', sqre, n, m, n, 0_ip_, d, e, vt, ldu, u, ldu, &
                u, ldu, work, info)
            END IF
            RETURN
          END IF
          inode = 1
          ndiml = inode + n
          ndimr = ndiml + n
          idxq = ndimr + n
          iwk = idxq + n
          ncc = 0
          nru = 0
          smlszp = smlsiz + 1
          vf = 1
          vl = vf + m
          nwork1 = vl + m
          nwork2 = nwork1 + smlszp*smlszp
          CALL DLASDT(n, nlvl, nd, iwork(inode), iwork(ndiml),              &
            iwork(ndimr), smlsiz)
          ndb1 = (nd+1)/2
          DO i = ndb1, nd
            i1 = i - 1
            ic = iwork(inode+i1)
            nl = iwork(ndiml+i1)
            nlp1 = nl + 1
            nr = iwork(ndimr+i1)
            nlf = ic - nl
            nrf = ic + 1
            idxqi = idxq + nlf - 2
            vfi = vf + nlf - 1
            vli = vl + nlf - 1
            sqrei = 1
            IF (icompq==0) THEN
              CALL DLASET('A', nlp1, nlp1, zero, one, work(nwork1), smlszp)
              CALL DLASDQ('U', sqrei, nl, nlp1, nru, ncc, d(nlf), e(nlf),   &
                work(nwork1), smlszp, work(nwork2), nl, work(nwork2), nl,   &
                work(nwork2), info)
              itemp = nwork1 + nl*smlszp
              CALL DCOPY(nlp1, work(nwork1), 1_ip_, work(vfi), 1_ip_)
              CALL DCOPY(nlp1, work(itemp), 1_ip_, work(vli), 1_ip_)
            ELSE
              CALL DLASET('A', nl, nl, zero, one, u(nlf,1), ldu)
              CALL DLASET('A', nlp1, nlp1, zero, one, vt(nlf,1), ldu)
              CALL DLASDQ('U', sqrei, nl, nlp1, nl, ncc, d(nlf), e(nlf),    &
                vt(nlf,1), ldu, u(nlf,1), ldu, u(nlf,1), ldu, work(nwork1), &
                info)
              CALL DCOPY(nlp1, vt(nlf,1), 1_ip_, work(vfi), 1_ip_)
              CALL DCOPY(nlp1, vt(nlf,nlp1), 1_ip_, work(vli), 1_ip_)
            END IF
            IF (info/=0) THEN
              RETURN
            END IF
            DO j = 1, nl
              iwork(idxqi+j) = j
            END DO
            IF ((i==nd) .AND. (sqre==0)) THEN
              sqrei = 0
            ELSE
              sqrei = 1
            END IF
            idxqi = idxqi + nlp1
            vfi = vfi + nlp1
            vli = vli + nlp1
            nrp1 = nr + sqrei
            IF (icompq==0) THEN
              CALL DLASET('A', nrp1, nrp1, zero, one, work(nwork1), smlszp)
              CALL DLASDQ('U', sqrei, nr, nrp1, nru, ncc, d(nrf), e(nrf),   &
                work(nwork1), smlszp, work(nwork2), nr, work(nwork2), nr,   &
                work(nwork2), info)
              itemp = nwork1 + (nrp1-1)*smlszp
              CALL DCOPY(nrp1, work(nwork1), 1_ip_, work(vfi), 1_ip_)
              CALL DCOPY(nrp1, work(itemp), 1_ip_, work(vli), 1_ip_)
            ELSE
              CALL DLASET('A', nr, nr, zero, one, u(nrf,1), ldu)
              CALL DLASET('A', nrp1, nrp1, zero, one, vt(nrf,1), ldu)
              CALL DLASDQ('U', sqrei, nr, nrp1, nr, ncc, d(nrf), e(nrf),    &
                vt(nrf,1), ldu, u(nrf,1), ldu, u(nrf,1), ldu, work(nwork1), &
                info)
              CALL DCOPY(nrp1, vt(nrf,1), 1_ip_, work(vfi), 1_ip_)
              CALL DCOPY(nrp1, vt(nrf,nrp1), 1_ip_, work(vli), 1_ip_)
            END IF
            IF (info/=0) THEN
              RETURN
            END IF
            DO j = 1, nr
              iwork(idxqi+j) = j
            END DO
          END DO
          j = 2**nlvl
          DO lvl = nlvl, 1_ip_, -1_ip_
            lvl2 = lvl*2 - 1
            IF (lvl==1) THEN
              lf = 1
              ll = 1
            ELSE
              lf = 2**(lvl-1)
              ll = 2*lf - 1
            END IF
            DO i = lf, ll
              im1 = i - 1
              ic = iwork(inode+im1)
              nl = iwork(ndiml+im1)
              nr = iwork(ndimr+im1)
              nlf = ic - nl
              nrf = ic + 1
              IF (i==ll) THEN
                sqrei = sqre
              ELSE
                sqrei = 1
              END IF
              vfi = vf + nlf - 1
              vli = vl + nlf - 1
              idxqi = idxq + nlf - 1
              alpha = d(ic)
              beta = e(ic)
              IF (icompq==0) THEN
                CALL DLASD6(icompq, nl, nr, sqrei, d(nlf), work(vfi),       &
                  work(vli), alpha, beta, iwork(idxqi), perm, givptr(1),    &
                  givcol, ldgcol, givnum, ldu, poles, difl, difr, z, k(1),  &
                  c(1), s(1), work(nwork1), iwork(iwk), info)
              ELSE
                j = j - 1
                CALL DLASD6(icompq, nl, nr, sqrei, d(nlf), work(vfi),       &
                  work(vli), alpha, beta, iwork(idxqi), perm(nlf,lvl),      &
                  givptr(j), givcol(nlf,lvl2), ldgcol, givnum(nlf,lvl2), ldu,&
                  poles(nlf,lvl2), difl(nlf,lvl), difr(nlf,lvl2), z(nlf,lvl),&
                  k(j), c(j), s(j), work(nwork1), iwork(iwk), info)
              END IF
              IF (info/=0) THEN
                RETURN
              END IF
            END DO
          END DO
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASDQ(uplo, sqre, n, ncvt, nru, ncc, d, e, vt, ldvt, u, &
          ldu, c, ldc, work, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, ldc, ldu, ldvt, n, ncc, ncvt, nru, sqre
          REAL(rp_) :: c(ldc, *), d(*), e(*), u(ldu, *), vt(ldvt, *),       &
            work(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          LOGICAL :: rotate
          INTEGER(ip_) :: i, isub, iuplo, j, np1, sqre1
          REAL(rp_) :: cs, r, smin, sn
          EXTERNAL :: DBDSQR, DLARTG, DLASR, DSWAP, XERBLA2
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          INTRINSIC :: MAX
          info = 0
          iuplo = 0
          IF (LSAME(uplo,'U')) iuplo = 1
          IF (LSAME(uplo,'L')) iuplo = 2
          IF (iuplo==0) THEN
            info = -1
          ELSE IF ((sqre<0) .OR. (sqre>1)) THEN
            info = -2
          ELSE IF (n<0) THEN
            info = -3
          ELSE IF (ncvt<0) THEN
            info = -4
          ELSE IF (nru<0) THEN
            info = -5
          ELSE IF (ncc<0) THEN
            info = -6
          ELSE IF ((ncvt==0 .AND. ldvt<1) .OR. (ncvt>0 .AND. ldvt<MAX(1,    &
            n))) THEN
            info = -10
          ELSE IF (ldu<MAX(1,nru)) THEN
            info = -12
          ELSE IF ((ncc==0 .AND. ldc<1) .OR. (ncc>0 .AND. ldc<MAX(1, n)))   &
            THEN
            info = -14
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('LASDQ', -info)
            RETURN
          END IF
          IF (n==0) RETURN
          rotate = (ncvt>0) .OR. (nru>0) .OR. (ncc>0)
          np1 = n + 1
          sqre1 = sqre
          IF ((iuplo==1) .AND. (sqre1==1)) THEN
            DO i = 1, n - 1
              CALL DLARTG(d(i), e(i), cs, sn, r)
              d(i) = r
              e(i) = sn*d(i+1)
              d(i+1) = cs*d(i+1)
              IF (rotate) THEN
                work(i) = cs
                work(n+i) = sn
              END IF
            END DO
            CALL DLARTG(d(n), e(n), cs, sn, r)
            d(n) = r
            e(n) = zero
            IF (rotate) THEN
              work(n) = cs
              work(n+n) = sn
            END IF
            iuplo = 2
            sqre1 = 0
            IF (ncvt>0) CALL DLASR('L', 'V', 'F', np1, ncvt, work(1),       &
              work(np1), vt, ldvt)
          END IF
          IF (iuplo==2) THEN
            DO i = 1, n - 1
              CALL DLARTG(d(i), e(i), cs, sn, r)
              d(i) = r
              e(i) = sn*d(i+1)
              d(i+1) = cs*d(i+1)
              IF (rotate) THEN
                work(i) = cs
                work(n+i) = sn
              END IF
            END DO
            IF (sqre1==1) THEN
              CALL DLARTG(d(n), e(n), cs, sn, r)
              d(n) = r
              IF (rotate) THEN
                work(n) = cs
                work(n+n) = sn
              END IF
            END IF
            IF (nru>0) THEN
              IF (sqre1==0) THEN
                CALL DLASR('R', 'V', 'F', nru, n, work(1), work(np1), u,    &
                  ldu)
              ELSE
                CALL DLASR('R', 'V', 'F', nru, np1, work(1), work(np1), u,  &
                  ldu)
              END IF
            END IF
            IF (ncc>0) THEN
              IF (sqre1==0) THEN
                CALL DLASR('L', 'V', 'F', n, ncc, work(1), work(np1), c,    &
                  ldc)
              ELSE
                CALL DLASR('L', 'V', 'F', np1, ncc, work(1), work(np1), c,  &
                  ldc)
              END IF
            END IF
          END IF
          CALL DBDSQR('U', n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c,    &
            ldc, work, info)
          DO i = 1, n
            isub = i
            smin = d(i)
            DO j = i + 1, n
              IF (d(j)<smin) THEN
                isub = j
                smin = d(j)
              END IF
            END DO
            IF (isub/=i) THEN
              d(isub) = d(i)
              d(i) = smin
              IF (ncvt>0) CALL DSWAP(ncvt, vt(isub,1), ldvt, vt(i,1), ldvt)
              IF (nru>0) CALL DSWAP(nru, u(1,isub), 1_ip_, u(1,i), 1_ip_)
              IF (ncc>0) CALL DSWAP(ncc, c(isub,1), ldc, c(i,1), ldc)
            END IF
          END DO
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASDT(n, lvl, nd, inode, ndiml, ndimr, msub)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: lvl, msub, n, nd
          INTEGER(ip_) :: inode(*), ndiml(*), ndimr(*)
          REAL(rp_) :: two
          PARAMETER (two=2.0_rp_)
          INTEGER(ip_) :: i, il, ir, llst, maxn, ncrnt, nlvl
          REAL(rp_) :: temp
          INTRINSIC :: REAL, INT, LOG, MAX
          maxn = MAX(1, n)
          temp = LOG(REAL(maxn,rp_)/REAL(msub+1,rp_))/LOG(two)
          lvl = INT(temp) + 1
          i = n/2
          inode(1) = i + 1
          ndiml(1) = i
          ndimr(1) = n - i - 1
          il = 0
          ir = 1
          llst = 1
          DO nlvl = 1, lvl - 1
            DO i = 0, llst - 1
              il = il + 2
              ir = ir + 2
              ncrnt = llst + i
              ndiml(il) = ndiml(ncrnt)/2
              ndimr(il) = ndiml(ncrnt) - ndiml(il) - 1
              inode(il) = inode(ncrnt) - ndimr(il) - 1
              ndiml(ir) = ndimr(ncrnt)/2
              ndimr(ir) = ndimr(ncrnt) - ndiml(ir) - 1
              inode(ir) = inode(ncrnt) + ndiml(ir) + 1
            END DO
            llst = llst*2
          END DO
          nd = llst*2 - 1
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASET(uplo, m, n, alpha, beta, a, lda)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: lda, m, n
          REAL(rp_) :: alpha, beta
          REAL(rp_) :: a(lda, *)
          INTEGER(ip_) :: i, j
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          INTRINSIC :: MIN
          IF (LSAME(uplo,'U')) THEN
            DO j = 2, n
              DO i = 1, MIN(j-1, m)
                a(i, j) = alpha
              END DO
            END DO
          ELSE IF (LSAME(uplo,'L')) THEN
            DO j = 1, MIN(m, n)
              DO i = j + 1, m
                a(i, j) = alpha
              END DO
            END DO
          ELSE
            DO j = 1, n
              DO i = 1, m
                a(i, j) = alpha
              END DO
            END DO
          END IF
          DO i = 1, MIN(m, n)
            a(i, i) = beta
          END DO
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASQ1(n, d, e, work, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, n
          REAL(rp_) :: d(*), e(*), work(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          INTEGER(ip_) :: i, iinfo
          REAL(rp_) :: eps, scale, safmin, sigmn, sigmx
          EXTERNAL :: DCOPY, DLAS2, DLASCL, DLASQ2, DLASRT,& 
            XERBLA2
          REAL(rp_) :: DLAMCH
          EXTERNAL :: DLAMCH
          INTRINSIC :: ABS, MAX, SQRT
          info = 0
          IF (n<0) THEN
            info = -1
            CALL XERBLA2('LASQ1', -info)
            RETURN
          ELSE IF (n==0) THEN
            RETURN
          ELSE IF (n==1) THEN
            d(1) = ABS(d(1))
            RETURN
          ELSE IF (n==2) THEN
            CALL DLAS2(d(1), e(1), d(2), sigmn, sigmx)
            d(1) = sigmx
            d(2) = sigmn
            RETURN
          END IF
          sigmx = zero
          DO i = 1, n - 1
            d(i) = ABS(d(i))
            sigmx = MAX(sigmx, ABS(e(i)))
          END DO
          d(n) = ABS(d(n))
          IF (sigmx==zero) THEN
            CALL DLASRT('D', n, d, iinfo)
            RETURN
          END IF
          DO i = 1, n
            sigmx = MAX(sigmx, d(i))
          END DO
          eps = DLAMCH('Precision')
          safmin = DLAMCH('Safe minimum')
          scale = SQRT(eps/safmin)
          CALL DCOPY(n, d, 1_ip_, work(1), 2_ip_)
          CALL DCOPY(n-1, e, 1_ip_, work(2), 2_ip_)
          CALL DLASCL('G', 0_ip_, 0_ip_, sigmx, scale, 2*n-1, 1_ip_, work,  &
            2*n-1, iinfo)
          DO i = 1, 2*n - 1
            work(i) = work(i)**2
          END DO
          work(2*n) = zero
          CALL DLASQ2(n, work, info)
          IF (info==0) THEN
            DO i = 1, n
              d(i) = SQRT(work(i))
            END DO
            CALL DLASCL('G', 0_ip_, 0_ip_, scale, sigmx, n, 1_ip_, d, n,    &
              iinfo)
          ELSE IF (info==2) THEN
            DO i = 1, n
              d(i) = SQRT(work(2*i-1))
              e(i) = SQRT(work(2*i))
            END DO
            CALL DLASCL('G', 0_ip_, 0_ip_, scale, sigmx, n, 1_ip_, d, n,    &
              iinfo)
            CALL DLASCL('G', 0_ip_, 0_ip_, scale, sigmx, n, 1_ip_, e, n,    &
              iinfo)
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASQ2(n, z, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, n
          REAL(rp_) :: z(*)
          REAL(rp_) :: cbias
          PARAMETER (cbias=1.50_rp_)
          REAL(rp_) :: zero, half, one, two, four, hundrd
          PARAMETER (zero=0.0_rp_, half=0.5_rp_, one=1.0_rp_, two=2.0_rp_,  &
            four=4.0_rp_, hundrd=100.0_rp_)
          LOGICAL :: ieee
          INTEGER(ip_) :: i0, i1, i4, iinfo, ipn4, iter, iwhila, iwhilb, k, &
            kmin, n0, n1, nbig, ndiv, nfail, pp, splt, ttype
          REAL(rp_) :: d, dee, deemin, desig, dmin, dmin1, dmin2, dn, dn1,  &
            dn2, e, emax, emin, eps, g, oldemn, qmax, qmin, s, safmin, sigma,&
            t, tau, temp, tol, tol2, trace, zmax, tempe, tempq
          EXTERNAL :: DLASQ3, DLASRT, XERBLA2
          INTEGER(ip_) :: ILAENV2
          REAL(rp_) :: DLAMCH
          EXTERNAL :: DLAMCH, ILAENV2
          INTRINSIC :: ABS, REAL, MAX, MIN, SQRT
          info = 0
          eps = DLAMCH('Precision')
          safmin = DLAMCH('Safe minimum')
          tol = eps*hundrd
          tol2 = tol**2
          IF (n<0) THEN
            info = -1
            CALL XERBLA2('LASQ2', 1_ip_)
            RETURN
          ELSE IF (n==0) THEN
            RETURN
          ELSE IF (n==1) THEN
            IF (z(1)<zero) THEN
              info = -201
              CALL XERBLA2('LASQ2', 2_ip_)
            END IF
            RETURN
          ELSE IF (n==2) THEN
            IF (z(1)<zero) THEN
              info = -201
              CALL XERBLA2('LASQ2', 2_ip_)
              RETURN
            ELSE IF (z(2)<zero) THEN
              info = -202
              CALL XERBLA2('LASQ2', 2_ip_)
              RETURN
            ELSE IF (z(3)<zero) THEN
              info = -203
              CALL XERBLA2('LASQ2', 2_ip_)
              RETURN
            ELSE IF (z(3)>z(1)) THEN
              d = z(3)
              z(3) = z(1)
              z(1) = d
            END IF
            z(5) = z(1) + z(2) + z(3)
            IF (z(2)>z(3)*tol2) THEN
              t = half*((z(1)-z(3))+z(2))
              s = z(3)*(z(2)/t)
              IF (s<=t) THEN
                s = z(3)*(z(2)/(t*(one+SQRT(one+s/t))))
              ELSE
                s = z(3)*(z(2)/(t+SQRT(t)*SQRT(t+s)))
              END IF
              t = z(1) + (s+z(2))
              z(3) = z(3)*(z(1)/t)
              z(1) = t
            END IF
            z(2) = z(3)
            z(6) = z(2) + z(1)
            RETURN
          END IF
          z(2*n) = zero
          emin = z(2)
          qmax = zero
          zmax = zero
          d = zero
          e = zero
          DO k = 1, 2*(n-1), 2
            IF (z(k)<zero) THEN
              info = -(200+k)
              CALL XERBLA2('LASQ2', 2_ip_)
              RETURN
            ELSE IF (z(k+1)<zero) THEN
              info = -(200+k+1)
              CALL XERBLA2('LASQ2', 2_ip_)
              RETURN
            END IF
            d = d + z(k)
            e = e + z(k+1)
            qmax = MAX(qmax, z(k))
            emin = MIN(emin, z(k+1))
            zmax = MAX(qmax, zmax, z(k+1))
          END DO
          IF (z(2*n-1)<zero) THEN
            info = -(200+2*n-1)
            CALL XERBLA2('LASQ2', 2_ip_)
            RETURN
          END IF
          d = d + z(2*n-1)
          qmax = MAX(qmax, z(2*n-1))
          zmax = MAX(qmax, zmax)
          IF (e==zero) THEN
            DO k = 2, n
              z(k) = z(2*k-1)
            END DO
            CALL DLASRT('D', n, z, iinfo)
            z(2*n-1) = d
            RETURN
          END IF
          trace = d + e
          IF (trace==zero) THEN
            z(2*n-1) = zero
            RETURN
          END IF
          ieee = (ILAENV2(10_ip_,'LASQ2','N',1_ip_,2_ip_,3_ip_,4_ip_)==1)
          DO k = 2*n, 2_ip_, -2
            z(2*k) = zero
            z(2*k-1) = z(k)
            z(2*k-2) = zero
            z(2*k-3) = z(k-1)
          END DO
          i0 = 1
          n0 = n
          IF (cbias*z(4*i0-3)<z(4*n0-3)) THEN
            ipn4 = 4*(i0+n0)
            DO i4 = 4*i0, 2*(i0+n0-1), 4
              temp = z(i4-3)
              z(i4-3) = z(ipn4-i4-3)
              z(ipn4-i4-3) = temp
              temp = z(i4-1)
              z(i4-1) = z(ipn4-i4-5)
              z(ipn4-i4-5) = temp
            END DO
          END IF
          pp = 0
          DO k = 1, 2
            d = z(4*n0+pp-3)
            DO i4 = 4*(n0-1) + pp, 4*i0 + pp, -4
              IF (z(i4-1)<=tol2*d) THEN
                z(i4-1) = -zero
                d = z(i4-3)
              ELSE
                d = z(i4-3)*(d/(d+z(i4-1)))
              END IF
            END DO
            emin = z(4*i0+pp+1)
            d = z(4*i0+pp-3)
            DO i4 = 4*i0 + pp, 4*(n0-1) + pp, 4
              z(i4-2*pp-2) = d + z(i4-1)
              IF (z(i4-1)<=tol2*d) THEN
                z(i4-1) = -zero
                z(i4-2*pp-2) = d
                z(i4-2*pp) = zero
                d = z(i4+1)
              ELSE IF (safmin*z(i4+1)<z(i4-2*pp-2) .AND.                    &
                safmin*z(i4-2*pp-2)<z(i4+1)) THEN
                temp = z(i4+1)/z(i4-2*pp-2)
                z(i4-2*pp) = z(i4-1)*temp
                d = d*temp
              ELSE
                z(i4-2*pp) = z(i4+1)*(z(i4-1)/z(i4-2*pp-2))
                d = z(i4+1)*(d/z(i4-2*pp-2))
              END IF
              emin = MIN(emin, z(i4-2*pp))
            END DO
            z(4*n0-pp-2) = d
            qmax = z(4*i0-pp-2)
            DO i4 = 4*i0 - pp + 2, 4*n0 - pp - 2, 4
              qmax = MAX(qmax, z(i4))
            END DO
            pp = 1 - pp
          END DO
          ttype = 0
          dmin1 = zero
          dmin2 = zero
          dn = zero
          dn1 = zero
          dn2 = zero
          g = zero
          tau = zero
          iter = 2
          nfail = 0
          ndiv = 2*(n0-i0)
          DO iwhila = 1, n + 1
            IF (n0<1) GO TO 170
            desig = zero
            IF (n0==n) THEN
              sigma = zero
            ELSE
              sigma = -z(4*n0-1)
            END IF
            IF (sigma<zero) THEN
              info = 1
              RETURN
            END IF
            emax = zero
            IF (n0>i0) THEN
              emin = ABS(z(4*n0-5))
            ELSE
              emin = zero
            END IF
            qmin = z(4*n0-3)
            qmax = qmin
            DO i4 = 4*n0, 8, -4
              IF (z(i4-5)<=zero) GO TO 100
              IF (qmin>=four*emax) THEN
                qmin = MIN(qmin, z(i4-3))
                emax = MAX(emax, z(i4-5))
              END IF
              qmax = MAX(qmax, z(i4-7)+z(i4-5))
              emin = MIN(emin, z(i4-5))
            END DO
            i4 = 4
 100        CONTINUE
            i0 = i4/4
            pp = 0
            IF (n0-i0>1) THEN
              dee = z(4*i0-3)
              deemin = dee
              kmin = i0
              DO i4 = 4*i0 + 1, 4*n0 - 3, 4
                dee = z(i4)*(dee/(dee+z(i4-2)))
                IF (dee<=deemin) THEN
                  deemin = dee
                  kmin = (i4+3)/4
                END IF
              END DO
              IF ((kmin-i0)*2<n0-kmin .AND. deemin<=half*z(4*n0-3)) THEN
                ipn4 = 4*(i0+n0)
                pp = 2
                DO i4 = 4*i0, 2*(i0+n0-1), 4
                  temp = z(i4-3)
                  z(i4-3) = z(ipn4-i4-3)
                  z(ipn4-i4-3) = temp
                  temp = z(i4-2)
                  z(i4-2) = z(ipn4-i4-2)
                  z(ipn4-i4-2) = temp
                  temp = z(i4-1)
                  z(i4-1) = z(ipn4-i4-5)
                  z(ipn4-i4-5) = temp
                  temp = z(i4)
                  z(i4) = z(ipn4-i4-4)
                  z(ipn4-i4-4) = temp
                END DO
              END IF
            END IF
            dmin = -MAX(zero, qmin-two*SQRT(qmin)*SQRT(emax))
            nbig = 100*(n0-i0+1)
            DO iwhilb = 1, nbig
              IF (i0>n0) GO TO 150
              CALL DLASQ3(i0, n0, z, pp, dmin, sigma, desig, qmax, nfail,   &
                iter, ndiv, ieee, ttype, dmin1, dmin2, dn, dn1, dn2, g, tau)
              pp = 1 - pp
              IF (pp==0 .AND. n0-i0>=3) THEN
                IF (z(4*n0)<=tol2*qmax .OR. z(4*n0-1)<=tol2*sigma) THEN
                  splt = i0 - 1
                  qmax = z(4*i0-3)
                  emin = z(4*i0-1)
                  oldemn = z(4*i0)
                  DO i4 = 4*i0, 4*(n0-3), 4
                    IF (z(i4)<=tol2*z(i4-3) .OR. z(i4-1)<=tol2*sigma) THEN
                      z(i4-1) = -sigma
                      splt = i4/4
                      qmax = zero
                      emin = z(i4+3)
                      oldemn = z(i4+4)
                    ELSE
                      qmax = MAX(qmax, z(i4+1))
                      emin = MIN(emin, z(i4-1))
                      oldemn = MIN(oldemn, z(i4))
                    END IF
                  END DO
                  z(4*n0-1) = emin
                  z(4*n0) = oldemn
                  i0 = splt + 1
                END IF
              END IF
            END DO
            info = 2
            i1 = i0
            n1 = n0
 145        CONTINUE
            tempq = z(4*i0-3)
            z(4*i0-3) = z(4*i0-3) + sigma
            DO k = i0 + 1, n0
              tempe = z(4*k-5)
              z(4*k-5) = z(4*k-5)*(tempq/z(4*k-7))
              tempq = z(4*k-3)
              z(4*k-3) = z(4*k-3) + sigma + tempe - z(4*k-5)
            END DO
            IF (i1>1) THEN
              n1 = i1 - 1
              DO WHILE ((i1>=2) .AND. (z(4*i1-5)>=zero))
                i1 = i1 - 1
              END DO
              sigma = -z(4*n1-1)
              GO TO 145
            END IF
            DO k = 1, n
              z(2*k-1) = z(4*k-3)
              IF (k<n0) THEN
                z(2*k) = z(4*k-1)
              ELSE
                z(2*k) = 0
              END IF
            END DO
            RETURN
 150        CONTINUE
          END DO
          info = 3
          RETURN
 170      CONTINUE
          DO k = 2, n
            z(k) = z(4*k-3)
          END DO
          CALL DLASRT('D', n, z, iinfo)
          e = zero
          DO k = n, 1_ip_, -1_ip_
            e = e + z(k)
          END DO
          z(2*n+1) = trace
          z(2*n+2) = e
          z(2*n+3) = REAL(iter,rp_)
          z(2*n+4) = REAL(ndiv,rp_)/REAL(n**2,rp_)
          z(2*n+5) = hundrd*REAL(nfail,rp_)/REAL(iter,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASQ3(i0, n0, z, pp, dmin, sigma, desig, qmax, nfail,   &
          iter, ndiv, ieee, ttype, dmin1, dmin2, dn, dn1, dn2, g, tau)
          USE BLAS_LAPACK_KINDS_precision
          LOGICAL :: ieee
          INTEGER(ip_) :: i0, iter, n0, ndiv, nfail, pp
          REAL(rp_) :: desig, dmin, dmin1, dmin2, dn, dn1, dn2, g, qmax,    &
            sigma, tau
          REAL(rp_) :: z(*)
          REAL(rp_) :: cbias
          PARAMETER (cbias=1.50_rp_)
          REAL(rp_) :: zero, qurtr, half, one, two, hundrd
          PARAMETER (zero=0.0_rp_, qurtr=0.250_rp_, half=0.5_rp_,           &
            one=1.0_rp_, two=2.0_rp_, hundrd=100.0_rp_)
          INTEGER(ip_) :: ipn4, j4, n0in, nn, ttype
          REAL(rp_) :: eps, s, t, temp, tol, tol2
          EXTERNAL :: DLASQ4, DLASQ5, DLASQ6
          REAL(rp_) :: DLAMCH
          LOGICAL :: LISNAN
          LOGICAL :: DISNAN
          EXTERNAL :: DISNAN, DLAMCH
          INTRINSIC :: ABS, MAX, MIN, SQRT
          n0in = n0
          eps = DLAMCH('Precision')
          tol = eps*hundrd
          tol2 = tol**2
 10       CONTINUE
          IF (n0<i0) RETURN
          IF (n0==i0) GO TO 20
          nn = 4*n0 + pp
          IF (n0==(i0+1)) GO TO 40
          IF (z(nn-5)>tol2*(sigma+z(nn-3)) .AND. z(nn-2*pp-4)>tol2*z(nn-7)  &
            ) GO TO 30
 20       CONTINUE
          z(4*n0-3) = z(4*n0+pp-3) + sigma
          n0 = n0 - 1
          GO TO 10
 30       CONTINUE
          IF (z(nn-9)>tol2*sigma .AND. z(nn-2*pp-8)>tol2*z(nn-11)) GO TO 50
 40       CONTINUE
          IF (z(nn-3)>z(nn-7)) THEN
            s = z(nn-3)
            z(nn-3) = z(nn-7)
            z(nn-7) = s
          END IF
          t = half*((z(nn-7)-z(nn-3))+z(nn-5))
          IF (z(nn-5)>z(nn-3)*tol2 .AND. t/=zero) THEN
            s = z(nn-3)*(z(nn-5)/t)
            IF (s<=t) THEN
              s = z(nn-3)*(z(nn-5)/(t*(one+SQRT(one+s/t))))
            ELSE
              s = z(nn-3)*(z(nn-5)/(t+SQRT(t)*SQRT(t+s)))
            END IF
            t = z(nn-7) + (s+z(nn-5))
            z(nn-3) = z(nn-3)*(z(nn-7)/t)
            z(nn-7) = t
          END IF
          z(4*n0-7) = z(nn-7) + sigma
          z(4*n0-3) = z(nn-3) + sigma
          n0 = n0 - 2
          GO TO 10
 50       CONTINUE
          IF (pp==2) pp = 0
          IF (dmin<=zero .OR. n0<n0in) THEN
            IF (cbias*z(4*i0+pp-3)<z(4*n0+pp-3)) THEN
              ipn4 = 4*(i0+n0)
              DO j4 = 4*i0, 2*(i0+n0-1), 4
                temp = z(j4-3)
                z(j4-3) = z(ipn4-j4-3)
                z(ipn4-j4-3) = temp
                temp = z(j4-2)
                z(j4-2) = z(ipn4-j4-2)
                z(ipn4-j4-2) = temp
                temp = z(j4-1)
                z(j4-1) = z(ipn4-j4-5)
                z(ipn4-j4-5) = temp
                temp = z(j4)
                z(j4) = z(ipn4-j4-4)
                z(ipn4-j4-4) = temp
              END DO
              IF (n0-i0<=4) THEN
                z(4*n0+pp-1) = z(4*i0+pp-1)
                z(4*n0-pp) = z(4*i0-pp)
              END IF
              dmin2 = MIN(dmin2, z(4*n0+pp-1))
              z(4*n0+pp-1) = MIN(z(4*n0+pp-1), z(4*i0+pp-1), z(4*i0+pp+3))
              z(4*n0-pp) = MIN(z(4*n0-pp), z(4*i0-pp), z(4*i0-pp+4))
              qmax = MAX(qmax, z(4*i0+pp-3), z(4*i0+pp+1))
              dmin = -zero
            END IF
          END IF
          CALL DLASQ4(i0, n0, z, pp, n0in, dmin, dmin1, dmin2, dn, dn1,     &
            dn2, tau, ttype, g)
 70       CONTINUE
          CALL DLASQ5(i0, n0, z, pp, tau, sigma, dmin, dmin1, dmin2, dn,    &
            dn1, dn2, ieee, eps)
          ndiv = ndiv + (n0-i0+2)
          iter = iter + 1
          LISNAN = DISNAN(dmin)
          IF (dmin>=zero .AND. dmin1>=zero) THEN
            GO TO 90
          ELSE IF (dmin<zero .AND. dmin1>zero .AND. z(4*(n0-                &
            1)-pp)<tol*(sigma+dn1) .AND. ABS(dn)<tol*sigma) THEN
            z(4*(n0-1)-pp+2) = zero
            dmin = zero
            GO TO 90
          ELSE IF (dmin<zero) THEN
            nfail = nfail + 1
            IF (ttype<-22) THEN
              tau = zero
            ELSE IF (dmin1>zero) THEN
              tau = (tau+dmin)*(one-two*eps)
              ttype = ttype - 11
            ELSE
              tau = qurtr*tau
              ttype = ttype - 12
            END IF
            GO TO 70
          ELSE IF (LISNAN) THEN
            IF (tau==zero) THEN
              GO TO 80
            ELSE
              tau = zero
              GO TO 70
            END IF
          ELSE
            GO TO 80
          END IF
 80       CONTINUE
          CALL DLASQ6(i0, n0, z, pp, dmin, dmin1, dmin2, dn, dn1, dn2)
          ndiv = ndiv + (n0-i0+2)
          iter = iter + 1
          tau = zero
 90       CONTINUE
          IF (tau<sigma) THEN
            desig = desig + tau
            t = sigma + desig
            desig = desig - (t-sigma)
          ELSE
            t = sigma + tau
            desig = sigma - (t-tau) + desig
          END IF
          sigma = t
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASQ4(i0, n0, z, pp, n0in, dmin, dmin1, dmin2, dn, dn1, &
          dn2, tau, ttype, g)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: i0, n0, n0in, pp, ttype
          REAL(rp_) :: dmin, dmin1, dmin2, dn, dn1, dn2, g, tau
          REAL(rp_) :: z(*)
          REAL(rp_) :: cnst1, cnst2, cnst3
          PARAMETER (cnst1=0.5630_rp_, cnst2=1.010_rp_, cnst3=1.050_rp_)
          REAL(rp_) :: qurtr, third, half, zero, one, two, hundrd
          PARAMETER (qurtr=0.250_rp_, third=0.3330_rp_, half=0.50_rp_,      &
            zero=0.0_rp_, one=1.0_rp_, two=2.0_rp_, hundrd=100.0_rp_)
          INTEGER(ip_) :: i4, nn, np
          REAL(rp_) :: a2, b1, b2, gam, gap1, gap2, s
          INTRINSIC :: MAX, MIN, SQRT
          IF (dmin<=zero) THEN
            tau = -dmin
            ttype = -1
            RETURN
          END IF
          nn = 4*n0 + pp
          IF (n0in==n0) THEN
            IF (dmin==dn .OR. dmin==dn1) THEN
              b1 = SQRT(z(nn-3))*SQRT(z(nn-5))
              b2 = SQRT(z(nn-7))*SQRT(z(nn-9))
              a2 = z(nn-7) + z(nn-5)
              IF (dmin==dn .AND. dmin1==dn1) THEN
                gap2 = dmin2 - a2 - dmin2*qurtr
                IF (gap2>zero .AND. gap2>b2) THEN
                  gap1 = a2 - dn - (b2/gap2)*b2
                ELSE
                  gap1 = a2 - dn - (b1+b2)
                END IF
                IF (gap1>zero .AND. gap1>b1) THEN
                  s = MAX(dn-(b1/gap1)*b1, half*dmin)
                  ttype = -2
                ELSE
                  s = zero
                  IF (dn>b1) s = dn - b1
                  IF (a2>(b1+b2)) s = MIN(s, a2-(b1+b2))
                  s = MAX(s, third*dmin)
                  ttype = -3
                END IF
              ELSE
                ttype = -4
                s = qurtr*dmin
                IF (dmin==dn) THEN
                  gam = dn
                  a2 = zero
                  IF (z(nn-5)>z(nn-7)) RETURN
                  b2 = z(nn-5)/z(nn-7)
                  np = nn - 9
                ELSE
                  np = nn - 2*pp
                  gam = dn1
                  IF (z(np-4)>z(np-2)) RETURN
                  a2 = z(np-4)/z(np-2)
                  IF (z(nn-9)>z(nn-11)) RETURN
                  b2 = z(nn-9)/z(nn-11)
                  np = nn - 13
                END IF
                a2 = a2 + b2
                DO i4 = np, 4*i0 - 1 + pp, -4
                  IF (b2==zero) GO TO 20
                  b1 = b2
                  IF (z(i4)>z(i4-2)) RETURN
                  b2 = b2*(z(i4)/z(i4-2))
                  a2 = a2 + b2
                  IF (hundrd*MAX(b2,b1)<a2 .OR. cnst1<a2) GO TO 20
                END DO
 20             CONTINUE
                a2 = cnst3*a2
                IF (a2<cnst1) s = gam*(one-SQRT(a2))/(one+a2)
              END IF
            ELSE IF (dmin==dn2) THEN
              ttype = -5
              s = qurtr*dmin
              np = nn - 2*pp
              b1 = z(np-2)
              b2 = z(np-6)
              gam = dn2
              IF (z(np-8)>b2 .OR. z(np-4)>b1) RETURN
              a2 = (z(np-8)/b2)*(one+z(np-4)/b1)
              IF (n0-i0>2) THEN
                b2 = z(nn-13)/z(nn-15)
                a2 = a2 + b2
                DO i4 = nn - 17, 4*i0 - 1 + pp, -4
                  IF (b2==zero) GO TO 40
                  b1 = b2
                  IF (z(i4)>z(i4-2)) RETURN
                  b2 = b2*(z(i4)/z(i4-2))
                  a2 = a2 + b2
                  IF (hundrd*MAX(b2,b1)<a2 .OR. cnst1<a2) GO TO 40
                END DO
 40             CONTINUE
                a2 = cnst3*a2
              END IF
              IF (a2<cnst1) s = gam*(one-SQRT(a2))/(one+a2)
            ELSE
              IF (ttype==-6) THEN
                g = g + third*(one-g)
              ELSE IF (ttype==-18) THEN
                g = qurtr*third
              ELSE
                g = qurtr
              END IF
              s = g*dmin
              ttype = -6
            END IF
          ELSE IF (n0in==(n0+1)) THEN
            IF (dmin1==dn1 .AND. dmin2==dn2) THEN
              ttype = -7
              s = third*dmin1
              IF (z(nn-5)>z(nn-7)) RETURN
              b1 = z(nn-5)/z(nn-7)
              b2 = b1
              IF (b2==zero) GO TO 60
              DO i4 = 4*n0 - 9 + pp, 4*i0 - 1 + pp, -4
                a2 = b1
                IF (z(i4)>z(i4-2)) RETURN
                b1 = b1*(z(i4)/z(i4-2))
                b2 = b2 + b1
                IF (hundrd*MAX(b1,a2)<b2) GO TO 60
              END DO
 60           CONTINUE
              b2 = SQRT(cnst3*b2)
              a2 = dmin1/(one+b2**2)
              gap2 = half*dmin2 - a2
              IF (gap2>zero .AND. gap2>b2*a2) THEN
                s = MAX(s, a2*(one-cnst2*a2*(b2/gap2)*b2))
              ELSE
                s = MAX(s, a2*(one-cnst2*b2))
                ttype = -8
              END IF
            ELSE
              s = qurtr*dmin1
              IF (dmin1==dn1) s = half*dmin1
              ttype = -9
            END IF
          ELSE IF (n0in==(n0+2)) THEN
            IF (dmin2==dn2 .AND. two*z(nn-5)<z(nn-7)) THEN
              ttype = -10
              s = third*dmin2
              IF (z(nn-5)>z(nn-7)) RETURN
              b1 = z(nn-5)/z(nn-7)
              b2 = b1
              IF (b2==zero) GO TO 80
              DO i4 = 4*n0 - 9 + pp, 4*i0 - 1 + pp, -4
                IF (z(i4)>z(i4-2)) RETURN
                b1 = b1*(z(i4)/z(i4-2))
                b2 = b2 + b1
                IF (hundrd*b1<b2) GO TO 80
              END DO
 80           CONTINUE
              b2 = SQRT(cnst3*b2)
              a2 = dmin2/(one+b2**2)
              gap2 = z(nn-7) + z(nn-9) - SQRT(z(nn-11))*SQRT(z(nn-9)) - a2
              IF (gap2>zero .AND. gap2>b2*a2) THEN
                s = MAX(s, a2*(one-cnst2*a2*(b2/gap2)*b2))
              ELSE
                s = MAX(s, a2*(one-cnst2*b2))
              END IF
            ELSE
              s = qurtr*dmin2
              ttype = -11
            END IF
          ELSE IF (n0in>(n0+2)) THEN
            s = zero
            ttype = -12
          END IF
          tau = s
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASQ5(i0, n0, z, pp, tau, sigma, dmin, dmin1, dmin2,    &
          dn, dnm1, dnm2, ieee, eps)
          USE BLAS_LAPACK_KINDS_precision
          LOGICAL :: ieee
          INTEGER(ip_) :: i0, n0, pp
          REAL(rp_) :: dmin, dmin1, dmin2, dn, dnm1, dnm2, tau, sigma, eps
          REAL(rp_) :: z(*)
          REAL(rp_) :: zero, half
          PARAMETER (zero=0.0_rp_, half=0.5)
          INTEGER(ip_) :: j4, j4p2
          REAL(rp_) :: d, emin, temp, dthresh
          INTRINSIC :: MIN
          IF ((n0-i0-1)<=0) RETURN
          dthresh = eps*(sigma+tau)
          IF (tau<dthresh*half) tau = zero
          IF (tau/=zero) THEN
            j4 = 4*i0 + pp - 3
            emin = z(j4+4)
            d = z(j4) - tau
            dmin = d
            dmin1 = -z(j4)
            IF (ieee) THEN
              IF (pp==0) THEN
                DO j4 = 4*i0, 4*(n0-3), 4
                  z(j4-2) = d + z(j4-1)
                  temp = z(j4+1)/z(j4-2)
                  d = d*temp - tau
                  dmin = MIN(dmin, d)
                  z(j4) = z(j4-1)*temp
                  emin = MIN(z(j4), emin)
                END DO
              ELSE
                DO j4 = 4*i0, 4*(n0-3), 4
                  z(j4-3) = d + z(j4)
                  temp = z(j4+2)/z(j4-3)
                  d = d*temp - tau
                  dmin = MIN(dmin, d)
                  z(j4-1) = z(j4)*temp
                  emin = MIN(z(j4-1), emin)
                END DO
              END IF
              dnm2 = d
              dmin2 = dmin
              j4 = 4*(n0-2) - pp
              j4p2 = j4 + 2*pp - 1
              z(j4-2) = dnm2 + z(j4p2)
              z(j4) = z(j4p2+2)*(z(j4p2)/z(j4-2))
              dnm1 = z(j4p2+2)*(dnm2/z(j4-2)) - tau
              dmin = MIN(dmin, dnm1)
              dmin1 = dmin
              j4 = j4 + 4
              j4p2 = j4 + 2*pp - 1
              z(j4-2) = dnm1 + z(j4p2)
              z(j4) = z(j4p2+2)*(z(j4p2)/z(j4-2))
              dn = z(j4p2+2)*(dnm1/z(j4-2)) - tau
              dmin = MIN(dmin, dn)
            ELSE
              IF (pp==0) THEN
                DO j4 = 4*i0, 4*(n0-3), 4
                  z(j4-2) = d + z(j4-1)
                  IF (d<zero) THEN
                    RETURN
                  ELSE
                    z(j4) = z(j4+1)*(z(j4-1)/z(j4-2))
                    d = z(j4+1)*(d/z(j4-2)) - tau
                  END IF
                  dmin = MIN(dmin, d)
                  emin = MIN(emin, z(j4))
                END DO
              ELSE
                DO j4 = 4*i0, 4*(n0-3), 4
                  z(j4-3) = d + z(j4)
                  IF (d<zero) THEN
                    RETURN
                  ELSE
                    z(j4-1) = z(j4+2)*(z(j4)/z(j4-3))
                    d = z(j4+2)*(d/z(j4-3)) - tau
                  END IF
                  dmin = MIN(dmin, d)
                  emin = MIN(emin, z(j4-1))
                END DO
              END IF
              dnm2 = d
              dmin2 = dmin
              j4 = 4*(n0-2) - pp
              j4p2 = j4 + 2*pp - 1
              z(j4-2) = dnm2 + z(j4p2)
              IF (dnm2<zero) THEN
                RETURN
              ELSE
                z(j4) = z(j4p2+2)*(z(j4p2)/z(j4-2))
                dnm1 = z(j4p2+2)*(dnm2/z(j4-2)) - tau
              END IF
              dmin = MIN(dmin, dnm1)
              dmin1 = dmin
              j4 = j4 + 4
              j4p2 = j4 + 2*pp - 1
              z(j4-2) = dnm1 + z(j4p2)
              IF (dnm1<zero) THEN
                RETURN
              ELSE
                z(j4) = z(j4p2+2)*(z(j4p2)/z(j4-2))
                dn = z(j4p2+2)*(dnm1/z(j4-2)) - tau
              END IF
              dmin = MIN(dmin, dn)
            END IF
          ELSE
            j4 = 4*i0 + pp - 3
            emin = z(j4+4)
            d = z(j4) - tau
            dmin = d
            dmin1 = -z(j4)
            IF (ieee) THEN
              IF (pp==0) THEN
                DO j4 = 4*i0, 4*(n0-3), 4
                  z(j4-2) = d + z(j4-1)
                  temp = z(j4+1)/z(j4-2)
                  d = d*temp - tau
                  IF (d<dthresh) d = zero
                  dmin = MIN(dmin, d)
                  z(j4) = z(j4-1)*temp
                  emin = MIN(z(j4), emin)
                END DO
              ELSE
                DO j4 = 4*i0, 4*(n0-3), 4
                  z(j4-3) = d + z(j4)
                  temp = z(j4+2)/z(j4-3)
                  d = d*temp - tau
                  IF (d<dthresh) d = zero
                  dmin = MIN(dmin, d)
                  z(j4-1) = z(j4)*temp
                  emin = MIN(z(j4-1), emin)
                END DO
              END IF
              dnm2 = d
              dmin2 = dmin
              j4 = 4*(n0-2) - pp
              j4p2 = j4 + 2*pp - 1
              z(j4-2) = dnm2 + z(j4p2)
              z(j4) = z(j4p2+2)*(z(j4p2)/z(j4-2))
              dnm1 = z(j4p2+2)*(dnm2/z(j4-2)) - tau
              dmin = MIN(dmin, dnm1)
              dmin1 = dmin
              j4 = j4 + 4
              j4p2 = j4 + 2*pp - 1
              z(j4-2) = dnm1 + z(j4p2)
              z(j4) = z(j4p2+2)*(z(j4p2)/z(j4-2))
              dn = z(j4p2+2)*(dnm1/z(j4-2)) - tau
              dmin = MIN(dmin, dn)
            ELSE
              IF (pp==0) THEN
                DO j4 = 4*i0, 4*(n0-3), 4
                  z(j4-2) = d + z(j4-1)
                  IF (d<zero) THEN
                    RETURN
                  ELSE
                    z(j4) = z(j4+1)*(z(j4-1)/z(j4-2))
                    d = z(j4+1)*(d/z(j4-2)) - tau
                  END IF
                  IF (d<dthresh) d = zero
                  dmin = MIN(dmin, d)
                  emin = MIN(emin, z(j4))
                END DO
              ELSE
                DO j4 = 4*i0, 4*(n0-3), 4
                  z(j4-3) = d + z(j4)
                  IF (d<zero) THEN
                    RETURN
                  ELSE
                    z(j4-1) = z(j4+2)*(z(j4)/z(j4-3))
                    d = z(j4+2)*(d/z(j4-3)) - tau
                  END IF
                  IF (d<dthresh) d = zero
                  dmin = MIN(dmin, d)
                  emin = MIN(emin, z(j4-1))
                END DO
              END IF
              dnm2 = d
              dmin2 = dmin
              j4 = 4*(n0-2) - pp
              j4p2 = j4 + 2*pp - 1
              z(j4-2) = dnm2 + z(j4p2)
              IF (dnm2<zero) THEN
                RETURN
              ELSE
                z(j4) = z(j4p2+2)*(z(j4p2)/z(j4-2))
                dnm1 = z(j4p2+2)*(dnm2/z(j4-2)) - tau
              END IF
              dmin = MIN(dmin, dnm1)
              dmin1 = dmin
              j4 = j4 + 4
              j4p2 = j4 + 2*pp - 1
              z(j4-2) = dnm1 + z(j4p2)
              IF (dnm1<zero) THEN
                RETURN
              ELSE
                z(j4) = z(j4p2+2)*(z(j4p2)/z(j4-2))
                dn = z(j4p2+2)*(dnm1/z(j4-2)) - tau
              END IF
              dmin = MIN(dmin, dn)
            END IF
          END IF
          z(j4+2) = dn
          z(4*n0-pp) = emin
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASQ6(i0, n0, z, pp, dmin, dmin1, dmin2, dn, dnm1,      &
          dnm2)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: i0, n0, pp
          REAL(rp_) :: dmin, dmin1, dmin2, dn, dnm1, dnm2
          REAL(rp_) :: z(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          INTEGER(ip_) :: j4, j4p2
          REAL(rp_) :: d, emin, safmin, temp
          REAL(rp_) :: DLAMCH
          EXTERNAL :: DLAMCH
          INTRINSIC :: MIN
          IF ((n0-i0-1)<=0) RETURN
          safmin = DLAMCH('Safe minimum')
          j4 = 4*i0 + pp - 3
          emin = z(j4+4)
          d = z(j4)
          dmin = d
          IF (pp==0) THEN
            DO j4 = 4*i0, 4*(n0-3), 4
              z(j4-2) = d + z(j4-1)
              IF (z(j4-2)==zero) THEN
                z(j4) = zero
                d = z(j4+1)
                dmin = d
                emin = zero
              ELSE IF (safmin*z(j4+1)<z(j4-2) .AND.                         &
                safmin*z(j4-2)<z(j4+1)) THEN
                temp = z(j4+1)/z(j4-2)
                z(j4) = z(j4-1)*temp
                d = d*temp
              ELSE
                z(j4) = z(j4+1)*(z(j4-1)/z(j4-2))
                d = z(j4+1)*(d/z(j4-2))
              END IF
              dmin = MIN(dmin, d)
              emin = MIN(emin, z(j4))
            END DO
          ELSE
            DO j4 = 4*i0, 4*(n0-3), 4
              z(j4-3) = d + z(j4)
              IF (z(j4-3)==zero) THEN
                z(j4-1) = zero
                d = z(j4+2)
                dmin = d
                emin = zero
              ELSE IF (safmin*z(j4+2)<z(j4-3) .AND.                         &
                safmin*z(j4-3)<z(j4+2)) THEN
                temp = z(j4+2)/z(j4-3)
                z(j4-1) = z(j4)*temp
                d = d*temp
              ELSE
                z(j4-1) = z(j4+2)*(z(j4)/z(j4-3))
                d = z(j4+2)*(d/z(j4-3))
              END IF
              dmin = MIN(dmin, d)
              emin = MIN(emin, z(j4-1))
            END DO
          END IF
          dnm2 = d
          dmin2 = dmin
          j4 = 4*(n0-2) - pp
          j4p2 = j4 + 2*pp - 1
          z(j4-2) = dnm2 + z(j4p2)
          IF (z(j4-2)==zero) THEN
            z(j4) = zero
            dnm1 = z(j4p2+2)
            dmin = dnm1
            emin = zero
          ELSE IF (safmin*z(j4p2+2)<z(j4-2) .AND.                           &
            safmin*z(j4-2)<z(j4p2+2)) THEN
            temp = z(j4p2+2)/z(j4-2)
            z(j4) = z(j4p2)*temp
            dnm1 = dnm2*temp
          ELSE
            z(j4) = z(j4p2+2)*(z(j4p2)/z(j4-2))
            dnm1 = z(j4p2+2)*(dnm2/z(j4-2))
          END IF
          dmin = MIN(dmin, dnm1)
          dmin1 = dmin
          j4 = j4 + 4
          j4p2 = j4 + 2*pp - 1
          z(j4-2) = dnm1 + z(j4p2)
          IF (z(j4-2)==zero) THEN
            z(j4) = zero
            dn = z(j4p2+2)
            dmin = dn
            emin = zero
          ELSE IF (safmin*z(j4p2+2)<z(j4-2) .AND.                           &
            safmin*z(j4-2)<z(j4p2+2)) THEN
            temp = z(j4p2+2)/z(j4-2)
            z(j4) = z(j4p2)*temp
            dn = dnm1*temp
          ELSE
            z(j4) = z(j4p2+2)*(z(j4p2)/z(j4-2))
            dn = z(j4p2+2)*(dnm1/z(j4-2))
          END IF
          dmin = MIN(dmin, dn)
          z(j4+2) = dn
          z(4*n0-pp) = emin
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASR(side, pivot, direct, m, n, c, s, a, lda)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: direct, pivot, side
          INTEGER(ip_) :: lda, m, n
          REAL(rp_) :: a(lda, *), c(*), s(*)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          INTEGER(ip_) :: i, info, j
          REAL(rp_) :: ctemp, stemp, temp
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          INTRINSIC :: MAX
          info = 0
          IF (.NOT. (LSAME(side,'L') .OR. LSAME(side,'R'))) THEN
            info = 1
          ELSE IF (.NOT. (LSAME(pivot,'V') .OR. LSAME(pivot, 'T') &
            .OR. LSAME(pivot,'B'))) THEN
            info = 2
          ELSE IF (.NOT. (LSAME(direct,'F') .OR. LSAME(direct,'B'))) &
            THEN
            info = 3
          ELSE IF (m<0) THEN
            info = 4
          ELSE IF (n<0) THEN
            info = 5
          ELSE IF (lda<MAX(1,m)) THEN
            info = 9
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('LASR ', info)
            RETURN
          END IF
          IF ((m==0) .OR. (n==0)) RETURN
          IF (LSAME(side,'L')) THEN
            IF (LSAME(pivot,'V')) THEN
              IF (LSAME(direct,'F')) THEN
                DO j = 1, m - 1
                  ctemp = c(j)
                  stemp = s(j)
                  IF ((ctemp/=one) .OR. (stemp/=zero)) THEN
                    DO i = 1, n
                      temp = a(j+1, i)
                      a(j+1, i) = ctemp*temp - stemp*a(j, i)
                      a(j, i) = stemp*temp + ctemp*a(j, i)
                    END DO
                  END IF
                END DO
              ELSE IF (LSAME(direct,'B')) THEN
                DO j = m - 1, 1_ip_, -1_ip_
                  ctemp = c(j)
                  stemp = s(j)
                  IF ((ctemp/=one) .OR. (stemp/=zero)) THEN
                    DO i = 1, n
                      temp = a(j+1, i)
                      a(j+1, i) = ctemp*temp - stemp*a(j, i)
                      a(j, i) = stemp*temp + ctemp*a(j, i)
                    END DO
                  END IF
                END DO
              END IF
            ELSE IF (LSAME(pivot,'T')) THEN
              IF (LSAME(direct,'F')) THEN
                DO j = 2, m
                  ctemp = c(j-1)
                  stemp = s(j-1)
                  IF ((ctemp/=one) .OR. (stemp/=zero)) THEN
                    DO i = 1, n
                      temp = a(j, i)
                      a(j, i) = ctemp*temp - stemp*a(1, i)
                      a(1, i) = stemp*temp + ctemp*a(1, i)
                    END DO
                  END IF
                END DO
              ELSE IF (LSAME(direct,'B')) THEN
                DO j = m, 2_ip_, -1_ip_
                  ctemp = c(j-1)
                  stemp = s(j-1)
                  IF ((ctemp/=one) .OR. (stemp/=zero)) THEN
                    DO i = 1, n
                      temp = a(j, i)
                      a(j, i) = ctemp*temp - stemp*a(1, i)
                      a(1, i) = stemp*temp + ctemp*a(1, i)
                    END DO
                  END IF
                END DO
              END IF
            ELSE IF (LSAME(pivot,'B')) THEN
              IF (LSAME(direct,'F')) THEN
                DO j = 1, m - 1
                  ctemp = c(j)
                  stemp = s(j)
                  IF ((ctemp/=one) .OR. (stemp/=zero)) THEN
                    DO i = 1, n
                      temp = a(j, i)
                      a(j, i) = stemp*a(m, i) + ctemp*temp
                      a(m, i) = ctemp*a(m, i) - stemp*temp
                    END DO
                  END IF
                END DO
              ELSE IF (LSAME(direct,'B')) THEN
                DO j = m - 1, 1_ip_, -1_ip_
                  ctemp = c(j)
                  stemp = s(j)
                  IF ((ctemp/=one) .OR. (stemp/=zero)) THEN
                    DO i = 1, n
                      temp = a(j, i)
                      a(j, i) = stemp*a(m, i) + ctemp*temp
                      a(m, i) = ctemp*a(m, i) - stemp*temp
                    END DO
                  END IF
                END DO
              END IF
            END IF
          ELSE IF (LSAME(side,'R')) THEN
            IF (LSAME(pivot,'V')) THEN
              IF (LSAME(direct,'F')) THEN
                DO j = 1, n - 1
                  ctemp = c(j)
                  stemp = s(j)
                  IF ((ctemp/=one) .OR. (stemp/=zero)) THEN
                    DO i = 1, m
                      temp = a(i, j+1)
                      a(i, j+1) = ctemp*temp - stemp*a(i, j)
                      a(i, j) = stemp*temp + ctemp*a(i, j)
                    END DO
                  END IF
                END DO
              ELSE IF (LSAME(direct,'B')) THEN
                DO j = n - 1, 1_ip_, -1_ip_
                  ctemp = c(j)
                  stemp = s(j)
                  IF ((ctemp/=one) .OR. (stemp/=zero)) THEN
                    DO i = 1, m
                      temp = a(i, j+1)
                      a(i, j+1) = ctemp*temp - stemp*a(i, j)
                      a(i, j) = stemp*temp + ctemp*a(i, j)
                    END DO
                  END IF
                END DO
              END IF
            ELSE IF (LSAME(pivot,'T')) THEN
              IF (LSAME(direct,'F')) THEN
                DO j = 2, n
                  ctemp = c(j-1)
                  stemp = s(j-1)
                  IF ((ctemp/=one) .OR. (stemp/=zero)) THEN
                    DO i = 1, m
                      temp = a(i, j)
                      a(i, j) = ctemp*temp - stemp*a(i, 1_ip_)
                      a(i, 1_ip_) = stemp*temp + ctemp*a(i, 1_ip_)
                    END DO
                  END IF
                END DO
              ELSE IF (LSAME(direct,'B')) THEN
                DO j = n, 2_ip_, -1_ip_
                  ctemp = c(j-1)
                  stemp = s(j-1)
                  IF ((ctemp/=one) .OR. (stemp/=zero)) THEN
                    DO i = 1, m
                      temp = a(i, j)
                      a(i, j) = ctemp*temp - stemp*a(i, 1_ip_)
                      a(i, 1_ip_) = stemp*temp + ctemp*a(i, 1_ip_)
                    END DO
                  END IF
                END DO
              END IF
            ELSE IF (LSAME(pivot,'B')) THEN
              IF (LSAME(direct,'F')) THEN
                DO j = 1, n - 1
                  ctemp = c(j)
                  stemp = s(j)
                  IF ((ctemp/=one) .OR. (stemp/=zero)) THEN
                    DO i = 1, m
                      temp = a(i, j)
                      a(i, j) = stemp*a(i, n) + ctemp*temp
                      a(i, n) = ctemp*a(i, n) - stemp*temp
                    END DO
                  END IF
                END DO
              ELSE IF (LSAME(direct,'B')) THEN
                DO j = n - 1, 1_ip_, -1_ip_
                  ctemp = c(j)
                  stemp = s(j)
                  IF ((ctemp/=one) .OR. (stemp/=zero)) THEN
                    DO i = 1, m
                      temp = a(i, j)
                      a(i, j) = stemp*a(i, n) + ctemp*temp
                      a(i, n) = ctemp*a(i, n) - stemp*temp
                    END DO
                  END IF
                END DO
              END IF
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASRT(id, n, d, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: id
          INTEGER(ip_) :: info, n
          REAL(rp_) :: d(*)
          INTEGER(ip_) :: select
          PARAMETER (select=20)
          INTEGER(ip_) :: dir, endd, i, j, start, stkpnt
          REAL(rp_) :: d1, d2, d3, dmnmx, tmp
          INTEGER(ip_) :: stack( 2_ip_, 32)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          info = 0
          dir = -1
          IF (LSAME(id,'D')) THEN
            dir = 0
          ELSE IF (LSAME(id,'I')) THEN
            dir = 1
          END IF
          IF (dir==-1) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('LASRT', -info)
            RETURN
          END IF
          IF (n<=1) RETURN
          stkpnt = 1
          stack(1, 1_ip_) = 1
          stack( 2_ip_, 1_ip_) = n
 10       CONTINUE
          start = stack(1, stkpnt)
          endd = stack( 2_ip_, stkpnt)
          stkpnt = stkpnt - 1
          IF (endd-start<=select .AND. endd-start>0) THEN
            IF (dir==0) THEN
              DO i = start + 1, endd
                DO j = i, start + 1, -1_ip_
                  IF (d(j)>d(j-1)) THEN
                    dmnmx = d(j)
                    d(j) = d(j-1)
                    d(j-1) = dmnmx
                  ELSE
                    GO TO 30
                  END IF
                END DO
 30           END DO
            ELSE
              DO i = start + 1, endd
                DO j = i, start + 1, -1_ip_
                  IF (d(j)<d(j-1)) THEN
                    dmnmx = d(j)
                    d(j) = d(j-1)
                    d(j-1) = dmnmx
                  ELSE
                    GO TO 50
                  END IF
                END DO
 50           END DO
            END IF
          ELSE IF (endd-start>select) THEN
            d1 = d(start)
            d2 = d(endd)
            i = (start+endd)/2
            d3 = d(i)
            IF (d1<d2) THEN
              IF (d3<d1) THEN
                dmnmx = d1
              ELSE IF (d3<d2) THEN
                dmnmx = d3
              ELSE
                dmnmx = d2
              END IF
            ELSE
              IF (d3<d2) THEN
                dmnmx = d2
              ELSE IF (d3<d1) THEN
                dmnmx = d3
              ELSE
                dmnmx = d1
              END IF
            END IF
            IF (dir==0) THEN
              i = start - 1
              j = endd + 1
 60           CONTINUE
 70           CONTINUE
              j = j - 1
              IF (d(j)<dmnmx) GO TO 70
 80           CONTINUE
              i = i + 1
              IF (d(i)>dmnmx) GO TO 80
              IF (i<j) THEN
                tmp = d(i)
                d(i) = d(j)
                d(j) = tmp
                GO TO 60
              END IF
              IF (j-start>endd-j-1) THEN
                stkpnt = stkpnt + 1
                stack(1, stkpnt) = start
                stack( 2_ip_, stkpnt) = j
                stkpnt = stkpnt + 1
                stack(1, stkpnt) = j + 1
                stack( 2_ip_, stkpnt) = endd
              ELSE
                stkpnt = stkpnt + 1
                stack(1, stkpnt) = j + 1
                stack( 2_ip_, stkpnt) = endd
                stkpnt = stkpnt + 1
                stack(1, stkpnt) = start
                stack( 2_ip_, stkpnt) = j
              END IF
            ELSE
              i = start - 1
              j = endd + 1
 90           CONTINUE
 100          CONTINUE
              j = j - 1
              IF (d(j)>dmnmx) GO TO 100
 110          CONTINUE
              i = i + 1
              IF (d(i)<dmnmx) GO TO 110
              IF (i<j) THEN
                tmp = d(i)
                d(i) = d(j)
                d(j) = tmp
                GO TO 90
              END IF
              IF (j-start>endd-j-1) THEN
                stkpnt = stkpnt + 1
                stack(1, stkpnt) = start
                stack( 2_ip_, stkpnt) = j
                stkpnt = stkpnt + 1
                stack(1, stkpnt) = j + 1
                stack( 2_ip_, stkpnt) = endd
              ELSE
                stkpnt = stkpnt + 1
                stack(1, stkpnt) = j + 1
                stack( 2_ip_, stkpnt) = endd
                stkpnt = stkpnt + 1
                stack(1, stkpnt) = start
                stack( 2_ip_, stkpnt) = j
              END IF
            END IF
          END IF
          IF (stkpnt>0) GO TO 10
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASSQ(n, x, incx, scale, sumsq)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: incx, n
          REAL(rp_) :: scale, sumsq
          REAL(rp_) :: x(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          INTEGER(ip_) :: ix
          REAL(rp_) :: absxi
          LOGICAL :: LISNAN
          LOGICAL :: DISNAN
          EXTERNAL :: DISNAN
          INTRINSIC :: ABS
          IF (n>0) THEN
            DO ix = 1, 1 + (n-1)*incx, incx
              absxi = ABS(x(ix))
              LISNAN = DISNAN(absxi)
              IF (absxi>zero .OR. LISNAN) THEN
                IF (scale<absxi) THEN
                  sumsq = 1 + sumsq*(scale/absxi)**2
                  scale = absxi
                ELSE
                  sumsq = sumsq + (absxi/scale)**2
                END IF
              END IF
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASV2(f, g, h, ssmin, ssmax, snr, csr, snl, csl)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: csl, csr, f, g, h, snl, snr, ssmax, ssmin
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          REAL(rp_) :: half
          PARAMETER (half=0.5_rp_)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          REAL(rp_) :: two
          PARAMETER (two=2.0_rp_)
          REAL(rp_) :: four
          PARAMETER (four=4.0_rp_)
          LOGICAL :: gasmal, swap
          INTEGER(ip_) :: pmax
          REAL(rp_) :: a, clt, crt, d, fa, ft, ga, gt, ha, ht, l, m, mm, r, &
            s, slt, srt, t, temp, tsign, tt
          INTRINSIC :: ABS, SIGN, SQRT
          REAL(rp_) :: DLAMCH
          EXTERNAL :: DLAMCH
          ft = f
          fa = ABS(ft)
          ht = h
          ha = ABS(h)
          pmax = 1
          swap = (ha>fa)
          IF (swap) THEN
            pmax = 3
            temp = ft
            ft = ht
            ht = temp
            temp = fa
            fa = ha
            ha = temp
          END IF
          gt = g
          ga = ABS(gt)
          IF (ga==zero) THEN
            ssmin = ha
            ssmax = fa
            clt = one
            crt = one
            slt = zero
            srt = zero
          ELSE
            gasmal = .TRUE.
            IF (ga>fa) THEN
              pmax = 2
              IF ((fa/ga)<DLAMCH('EPS')) THEN
                gasmal = .FALSE.
                ssmax = ga
                IF (ha>one) THEN
                  ssmin = fa/(ga/ha)
                ELSE
                  ssmin = (fa/ga)*ha
                END IF
                clt = one
                slt = ht/gt
                srt = one
                crt = ft/gt
              END IF
            END IF
            IF (gasmal) THEN
              d = fa - ha
              IF (d==fa) THEN
                l = one
              ELSE
                l = d/fa
              END IF
              m = gt/ft
              t = two - l
              mm = m*m
              tt = t*t
              s = SQRT(tt+mm)
              IF (l==zero) THEN
                r = ABS(m)
              ELSE
                r = SQRT(l*l+mm)
              END IF
              a = half*(s+r)
              ssmin = ha/a
              ssmax = fa*a
              IF (mm==zero) THEN
                IF (l==zero) THEN
                  t = SIGN(two, ft)*SIGN(one, gt)
                ELSE
                  t = gt/SIGN(d, ft) + m/t
                END IF
              ELSE
                t = (m/(s+t)+m/(r+l))*(one+a)
              END IF
              l = SQRT(t*t+four)
              crt = two/l
              srt = t/l
              clt = (crt+srt*m)/a
              slt = (ht/ft)*srt/a
            END IF
          END IF
          IF (swap) THEN
            csl = srt
            snl = crt
            csr = slt
            snr = clt
          ELSE
            csl = clt
            snl = slt
            csr = crt
            snr = srt
          END IF
          IF (pmax==1) tsign = SIGN(one, csr)*SIGN(one, csl)*SIGN(one, f)
          IF (pmax==2) tsign = SIGN(one, snr)*SIGN(one, csl)*SIGN(one, g)
          IF (pmax==3) tsign = SIGN(one, snr)*SIGN(one, snl)*SIGN(one, h)
          ssmax = SIGN(ssmax, tsign)
          ssmin = SIGN(ssmin, tsign*SIGN(one,f)*SIGN(one,h))
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASWP(n, a, lda, k1, k2, ipiv, incx)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: incx, k1, k2, lda, n
          INTEGER(ip_) :: ipiv(*)
          REAL(rp_) :: a(lda, *)
          INTEGER(ip_) :: i, i1, i2, inc, ip, ix, ix0, j, k, n32
          REAL(rp_) :: temp
          IF (incx>0) THEN
            ix0 = k1
            i1 = k1
            i2 = k2
            inc = 1
          ELSE IF (incx<0) THEN
            ix0 = k1 + (k1-k2)*incx
            i1 = k2
            i2 = k1
            inc = -1
          ELSE
            RETURN
          END IF
          n32 = (n/32)*32
          IF (n32/=0) THEN
            DO j = 1, n32, 32
              ix = ix0
              DO i = i1, i2, inc
                ip = ipiv(ix)
                IF (ip/=i) THEN
                  DO k = j, j + 31
                    temp = a(i, k)
                    a(i, k) = a(ip, k)
                    a(ip, k) = temp
                  END DO
                END IF
                ix = ix + incx
              END DO
            END DO
          END IF
          IF (n32/=n) THEN
            n32 = n32 + 1
            ix = ix0
            DO i = i1, i2, inc
              ip = ipiv(ix)
              IF (ip/=i) THEN
                DO k = n32, n
                  temp = a(i, k)
                  a(i, k) = a(ip, k)
                  a(ip, k) = temp
                END DO
              END IF
              ix = ix + incx
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASY2(ltranl, ltranr, isgn, n1, n2, tl, ldtl, tr, ldtr, &
          b, ldb, scale, x, ldx, xnorm, info)
          USE BLAS_LAPACK_KINDS_precision
          LOGICAL :: ltranl, ltranr
          INTEGER(ip_) :: info, isgn, ldb, ldtl, ldtr, ldx, n1, n2
          REAL(rp_) :: scale, xnorm
          REAL(rp_) :: b(ldb, *), tl(ldtl, *), tr(ldtr, *), x(ldx, *)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          REAL(rp_) :: two, half, eight
          PARAMETER (two=2.0_rp_, half=0.5_rp_, eight=8.0_rp_)
          LOGICAL :: bswap, xswap
          INTEGER(ip_) :: i, ip, ipiv, ipsv, j, jp, jpsv, k
          REAL(rp_) :: bet, eps, gam, l21, sgn, smin, smlnum, tau1, temp,   &
            u11, u12, u22, xmax
          LOGICAL :: bswpiv(4), xswpiv(4)
          INTEGER(ip_) :: jpiv(4), locl21(4), locu12(4), locu22(4)
          REAL(rp_) :: btmp(4), t16( 4_ip_, 4_ip_), tmp(4), x2(2)
          INTEGER(ip_) :: IDAMAX
          REAL(rp_) :: DLAMCH
          EXTERNAL :: IDAMAX, DLAMCH
          EXTERNAL :: DCOPY, DSWAP
          INTRINSIC :: ABS, MAX
          DATA locu12/3, 4_ip_, 1_ip_, 2/, locl21/2, 1_ip_, 4_ip_, 3/,      &
            locu22/4, 3_ip_, 2_ip_, 1/
          DATA xswpiv/.FALSE., .FALSE., .TRUE., .TRUE./
          DATA bswpiv/.FALSE., .TRUE., .FALSE., .TRUE./
          info = 0
          IF (n1==0 .OR. n2==0) RETURN
          eps = DLAMCH('P')
          smlnum = DLAMCH('S')/eps
          sgn = REAL(isgn,rp_)
          k = n1 + n1 + n2 - 2
          GO TO (10, 20, 30, 50) k
 10       CONTINUE
          tau1 = tl(1, 1_ip_) + sgn*tr(1, 1_ip_)
          bet = ABS(tau1)
          IF (bet<=smlnum) THEN
            tau1 = smlnum
            bet = smlnum
            info = 1
          END IF
          scale = one
          gam = ABS(b(1,1))
          IF (smlnum*gam>bet) scale = one/gam
          x(1, 1_ip_) = (b(1,1)*scale)/tau1
          xnorm = ABS(x(1,1))
          RETURN
 20       CONTINUE
          smin = MAX(eps*MAX(ABS(tl(1,1)),ABS(tr(1,1)),ABS(tr(1,2)),ABS(    &
            tr(2,1)),ABS(tr(2,2))), smlnum)
          tmp(1) = tl(1, 1_ip_) + sgn*tr(1, 1_ip_)
          tmp(4) = tl(1, 1_ip_) + sgn*tr( 2_ip_, 2_ip_)
          IF (ltranr) THEN
            tmp(2) = sgn*tr( 2_ip_, 1_ip_)
            tmp(3) = sgn*tr(1, 2_ip_)
          ELSE
            tmp(2) = sgn*tr(1, 2_ip_)
            tmp(3) = sgn*tr( 2_ip_, 1_ip_)
          END IF
          btmp(1) = b(1, 1_ip_)
          btmp(2) = b(1, 2_ip_)
          GO TO 40
 30       CONTINUE
          smin = MAX(eps*MAX(ABS(tr(1,1)),ABS(tl(1,1)),ABS(tl(1,2)),ABS(    &
            tl(2,1)),ABS(tl(2,2))), smlnum)
          tmp(1) = tl(1, 1_ip_) + sgn*tr(1, 1_ip_)
          tmp(4) = tl( 2_ip_, 2_ip_) + sgn*tr(1, 1_ip_)
          IF (ltranl) THEN
            tmp(2) = tl(1, 2_ip_)
            tmp(3) = tl( 2_ip_, 1_ip_)
          ELSE
            tmp(2) = tl( 2_ip_, 1_ip_)
            tmp(3) = tl(1, 2_ip_)
          END IF
          btmp(1) = b(1, 1_ip_)
          btmp(2) = b( 2_ip_, 1_ip_)
 40       CONTINUE
          ipiv = IDAMAX( 4_ip_, tmp, 1_ip_)
          u11 = tmp(ipiv)
          IF (ABS(u11)<=smin) THEN
            info = 1
            u11 = smin
          END IF
          u12 = tmp(locu12(ipiv))
          l21 = tmp(locl21(ipiv))/u11
          u22 = tmp(locu22(ipiv)) - u12*l21
          xswap = xswpiv(ipiv)
          bswap = bswpiv(ipiv)
          IF (ABS(u22)<=smin) THEN
            info = 1
            u22 = smin
          END IF
          IF (bswap) THEN
            temp = btmp(2)
            btmp(2) = btmp(1) - l21*temp
            btmp(1) = temp
          ELSE
            btmp(2) = btmp(2) - l21*btmp(1)
          END IF
          scale = one
          IF ((two*smlnum)*ABS(btmp(2))>ABS(u22) .OR. (two*smlnum)*ABS(     &
            btmp(1))>ABS(u11)) THEN
            scale = half/MAX(ABS(btmp(1)), ABS(btmp(2)))
            btmp(1) = btmp(1)*scale
            btmp(2) = btmp(2)*scale
          END IF
          x2(2) = btmp(2)/u22
          x2(1) = btmp(1)/u11 - (u12/u11)*x2(2)
          IF (xswap) THEN
            temp = x2(2)
            x2(2) = x2(1)
            x2(1) = temp
          END IF
          x(1, 1_ip_) = x2(1)
          IF (n1==1) THEN
            x(1, 2_ip_) = x2(2)
            xnorm = ABS(x(1,1)) + ABS(x(1,2))
          ELSE
            x( 2_ip_, 1_ip_) = x2(2)
            xnorm = MAX(ABS(x(1,1)), ABS(x(2,1)))
          END IF
          RETURN
 50       CONTINUE
          smin = MAX(ABS(tr(1,1)), ABS(tr(1,2)), ABS(tr(2,1)), ABS(tr(      &
            2_ip_, 2_ip_)))
          smin = MAX(smin, ABS(tl(1,1)), ABS(tl(1,2)), ABS(tl( 2_ip_,       &
            1_ip_)), ABS(tl(2,2)))
          smin = MAX(eps*smin, smlnum)
          btmp(1) = zero
          CALL DCOPY( 16_ip_, btmp, 0_ip_, t16, 1_ip_)
          t16(1, 1_ip_) = tl(1, 1_ip_) + sgn*tr(1, 1_ip_)
          t16( 2_ip_, 2_ip_) = tl( 2_ip_, 2_ip_) + sgn*tr(1, 1_ip_)
          t16( 3_ip_, 3_ip_) = tl(1, 1_ip_) + sgn*tr( 2_ip_, 2_ip_)
          t16( 4_ip_, 4_ip_) = tl( 2_ip_, 2_ip_) + sgn*tr( 2_ip_, 2_ip_)
          IF (ltranl) THEN
            t16(1, 2_ip_) = tl( 2_ip_, 1_ip_)
            t16( 2_ip_, 1_ip_) = tl(1, 2_ip_)
            t16( 3_ip_, 4_ip_) = tl( 2_ip_, 1_ip_)
            t16( 4_ip_, 3_ip_) = tl(1, 2_ip_)
          ELSE
            t16(1, 2_ip_) = tl(1, 2_ip_)
            t16( 2_ip_, 1_ip_) = tl( 2_ip_, 1_ip_)
            t16( 3_ip_, 4_ip_) = tl(1, 2_ip_)
            t16( 4_ip_, 3_ip_) = tl( 2_ip_, 1_ip_)
          END IF
          IF (ltranr) THEN
            t16(1, 3_ip_) = sgn*tr(1, 2_ip_)
            t16( 2_ip_, 4_ip_) = sgn*tr(1, 2_ip_)
            t16( 3_ip_, 1_ip_) = sgn*tr( 2_ip_, 1_ip_)
            t16( 4_ip_, 2_ip_) = sgn*tr( 2_ip_, 1_ip_)
          ELSE
            t16(1, 3_ip_) = sgn*tr( 2_ip_, 1_ip_)
            t16( 2_ip_, 4_ip_) = sgn*tr( 2_ip_, 1_ip_)
            t16( 3_ip_, 1_ip_) = sgn*tr(1, 2_ip_)
            t16( 4_ip_, 2_ip_) = sgn*tr(1, 2_ip_)
          END IF
          btmp(1) = b(1, 1_ip_)
          btmp(2) = b( 2_ip_, 1_ip_)
          btmp(3) = b(1, 2_ip_)
          btmp(4) = b( 2_ip_, 2_ip_)
          DO i = 1, 3
            xmax = zero
            DO ip = i, 4
              DO jp = i, 4
                IF (ABS(t16(ip,jp))>=xmax) THEN
                  xmax = ABS(t16(ip,jp))
                  ipsv = ip
                  jpsv = jp
                END IF
              END DO
            END DO
            IF (ipsv/=i) THEN
              CALL DSWAP( 4_ip_, t16(ipsv,1), 4_ip_, t16(i,1), 4_ip_)
              temp = btmp(i)
              btmp(i) = btmp(ipsv)
              btmp(ipsv) = temp
            END IF
            IF (jpsv/=i) CALL DSWAP( 4_ip_, t16(1,jpsv), 1_ip_, t16(1,i),   &
              1_ip_)
            jpiv(i) = jpsv
            IF (ABS(t16(i,i))<smin) THEN
              info = 1
              t16(i, i) = smin
            END IF
            DO j = i + 1, 4
              t16(j, i) = t16(j, i)/t16(i, i)
              btmp(j) = btmp(j) - t16(j, i)*btmp(i)
              DO k = i + 1, 4
                t16(j, k) = t16(j, k) - t16(j, i)*t16(i, k)
              END DO
            END DO
          END DO
          IF (ABS(t16(4,4))<smin) THEN
            info = 1
            t16( 4_ip_, 4_ip_) = smin
          END IF
          scale = one
          IF ((eight*smlnum)*ABS(btmp(1))>ABS(t16(1, 1_ip_)) .OR.           &
            (eight*smlnum)*ABS(btmp(2))>ABS(t16( 2_ip_, 2_ip_)) .OR.        &
            (eight*smlnum)*ABS(btmp(3))>ABS(t16( 3_ip_, 3_ip_)) .OR.        &
            (eight*smlnum)*ABS(btmp(4))>ABS(t16(4,4))) THEN
            scale = (one/eight)/MAX(ABS(btmp(1)), ABS(btmp(2)), ABS(btmp(3  &
              )), ABS(btmp(4)))
            btmp(1) = btmp(1)*scale
            btmp(2) = btmp(2)*scale
            btmp(3) = btmp(3)*scale
            btmp(4) = btmp(4)*scale
          END IF
          DO i = 1, 4
            k = 5 - i
            temp = one/t16(k, k)
            tmp(k) = btmp(k)*temp
            DO j = k + 1, 4
              tmp(k) = tmp(k) - (temp*t16(k,j))*tmp(j)
            END DO
          END DO
          DO i = 1, 3
            IF (jpiv(4-i)/=4-i) THEN
              temp = tmp(4-i)
              tmp(4-i) = tmp(jpiv(4-i))
              tmp(jpiv(4-i)) = temp
            END IF
          END DO
          x(1, 1_ip_) = tmp(1)
          x( 2_ip_, 1_ip_) = tmp(2)
          x(1, 2_ip_) = tmp(3)
          x( 2_ip_, 2_ip_) = tmp(4)
          xnorm = MAX(ABS(tmp(1))+ABS(tmp(3)), ABS(tmp(2))+ABS(tmp(4)))
          RETURN
        END SUBROUTINE

        SUBROUTINE DLASYF(uplo, n, nb, kb, a, lda, ipiv, w, ldw, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, kb, lda, ldw, n, nb
          INTEGER(ip_) :: ipiv(*)
          REAL(rp_) :: a(lda, *), w(ldw, *)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          REAL(rp_) :: eight, sevten
          PARAMETER (eight=8.0_rp_, sevten=17.0_rp_)
          INTEGER(ip_) :: imax, j, jb, jj, jmax, jp, k, kk, kkw, kp, kstep, &
            kw
          REAL(rp_) :: absakk, alpha, colmax, d11, d21, d22, r1, rowmax, t
          LOGICAL :: LSAME
          INTEGER(ip_) :: IDAMAX
          EXTERNAL :: LSAME, IDAMAX
          EXTERNAL :: DCOPY, DGEMM, DGEMV, DSCAL, DSWAP
          INTRINSIC :: ABS, MAX, MIN, SQRT
          info = 0
          alpha = (one+SQRT(sevten))/eight
          IF (LSAME(uplo,'U')) THEN
            k = n
 10         CONTINUE
            kw = nb + k - n
            IF ((k<=n-nb+1 .AND. nb<n) .OR. k<1) GO TO 30
            CALL DCOPY(k, a(1,k), 1_ip_, w(1,kw), 1_ip_)
            IF (k<n) CALL DGEMV('No transpose', k, n-k, -one, a(1,k+1),     &
              lda, w(k,kw+1), ldw, one, w(1,kw), 1_ip_)
            kstep = 1
            absakk = ABS(w(k,kw))
            IF (k>1) THEN
              imax = IDAMAX(k-1, w(1,kw), 1_ip_)
              colmax = ABS(w(imax,kw))
            ELSE
              colmax = zero
            END IF
            IF (MAX(absakk,colmax)==zero) THEN
              IF (info==0) info = k
              kp = k
            ELSE
              IF (absakk>=alpha*colmax) THEN
                kp = k
              ELSE
                CALL DCOPY(imax, a(1,imax), 1_ip_, w(1,kw-1), 1_ip_)
                CALL DCOPY(k-imax, a(imax,imax+1), lda, w(imax+1,kw-1),     &
                  1_ip_)
                IF (k<n) CALL DGEMV('No transpose', k, n-k, -one, a(1,k+1), &
                  lda, w(imax,kw+1), ldw, one, w(1,kw-1), 1_ip_)
                jmax = imax + IDAMAX(k-imax, w(imax+1,kw-1), 1_ip_)
                rowmax = ABS(w(jmax,kw-1))
                IF (imax>1) THEN
                  jmax = IDAMAX(imax-1, w(1,kw-1), 1_ip_)
                  rowmax = MAX(rowmax, ABS(w(jmax,kw-1)))
                END IF
                IF (absakk>=alpha*colmax*(colmax/rowmax)) THEN
                  kp = k
                ELSE IF (ABS(w(imax,kw-1))>=alpha*rowmax) THEN
                  kp = imax
                  CALL DCOPY(k, w(1,kw-1), 1_ip_, w(1,kw), 1_ip_)
                ELSE
                  kp = imax
                  kstep = 2
                END IF
              END IF
              kk = k - kstep + 1
              kkw = nb + kk - n
              IF (kp/=kk) THEN
                a(kp, kp) = a(kk, kk)
                CALL DCOPY(kk-1-kp, a(kp+1,kk), 1_ip_, a(kp,kp+1), lda)
                IF (kp>1) CALL DCOPY(kp-1, a(1,kk), 1_ip_, a(1,kp), 1_ip_)
                IF (k<n) CALL DSWAP(n-k, a(kk,k+1), lda, a(kp,k+1), lda)
                CALL DSWAP(n-kk+1, w(kk,kkw), ldw, w(kp,kkw), ldw)
              END IF
              IF (kstep==1) THEN
                CALL DCOPY(k, w(1,kw), 1_ip_, a(1,k), 1_ip_)
                r1 = one/a(k, k)
                CALL DSCAL(k-1, r1, a(1,k), 1_ip_)
              ELSE
                IF (k>2) THEN
                  d21 = w(k-1, kw)
                  d11 = w(k, kw)/d21
                  d22 = w(k-1, kw-1)/d21
                  t = one/(d11*d22-one)
                  d21 = t/d21
                  DO j = 1, k - 2
                    a(j, k-1) = d21*(d11*w(j,kw-1)-w(j,kw))
                    a(j, k) = d21*(d22*w(j,kw)-w(j,kw-1))
                  END DO
                END IF
                a(k-1, k-1) = w(k-1, kw-1)
                a(k-1, k) = w(k-1, kw)
                a(k, k) = w(k, kw)
              END IF
            END IF
            IF (kstep==1) THEN
              ipiv(k) = kp
            ELSE
              ipiv(k) = -kp
              ipiv(k-1) = -kp
            END IF
            k = k - kstep
            GO TO 10
 30         CONTINUE
            DO j = ((k-1)/nb)*nb + 1, 1_ip_, -nb
              jb = MIN(nb, k-j+1)
              DO jj = j, j + jb - 1
                CALL DGEMV('No transpose', jj-j+1, n-k, -one, a(j,k+1),     &
                  lda, w(jj,kw+1), ldw, one, a(j,jj), 1_ip_)
              END DO
              CALL DGEMM('No transpose', 'Transpose', j-1, jb, n-k, -one,   &
                a(1,k+1), lda, w(j,kw+1), ldw, one, a(1,j), lda)
            END DO
            j = k + 1
 60         CONTINUE
            jj = j
            jp = ipiv(j)
            IF (jp<0) THEN
              jp = -jp
              j = j + 1
            END IF
            j = j + 1
            IF (jp/=jj .AND. j<=n) CALL DSWAP(n-j+1, a(jp,j), lda, a(jj,j), &
              lda)
            IF (j<n) GO TO 60
            kb = n - k
          ELSE
            k = 1
 70         CONTINUE
            IF ((k>=nb .AND. nb<n) .OR. k>n) GO TO 90
            CALL DCOPY(n-k+1, a(k,k), 1_ip_, w(k,k), 1_ip_)
            CALL DGEMV('No transpose', n-k+1, k-1, -one, a(k,1), lda, w(k,  &
              1), ldw, one, w(k,k), 1_ip_)
            kstep = 1
            absakk = ABS(w(k,k))
            IF (k<n) THEN
              imax = k + IDAMAX(n-k, w(k+1,k), 1_ip_)
              colmax = ABS(w(imax,k))
            ELSE
              colmax = zero
            END IF
            IF (MAX(absakk,colmax)==zero) THEN
              IF (info==0) info = k
              kp = k
            ELSE
              IF (absakk>=alpha*colmax) THEN
                kp = k
              ELSE
                CALL DCOPY(imax-k, a(imax,k), lda, w(k,k+1), 1_ip_)
                CALL DCOPY(n-imax+1, a(imax,imax), 1_ip_, w(imax,k+1),      &
                  1_ip_)
                CALL DGEMV('No transpose', n-k+1, k-1, -one, a(k,1), lda,   &
                  w(imax,1), ldw, one, w(k,k+1), 1_ip_)
                jmax = k - 1 + IDAMAX(imax-k, w(k,k+1), 1_ip_)
                rowmax = ABS(w(jmax,k+1))
                IF (imax<n) THEN
                  jmax = imax + IDAMAX(n-imax, w(imax+1,k+1), 1_ip_)
                  rowmax = MAX(rowmax, ABS(w(jmax,k+1)))
                END IF
                IF (absakk>=alpha*colmax*(colmax/rowmax)) THEN
                  kp = k
                ELSE IF (ABS(w(imax,k+1))>=alpha*rowmax) THEN
                  kp = imax
                  CALL DCOPY(n-k+1, w(k,k+1), 1_ip_, w(k,k), 1_ip_)
                ELSE
                  kp = imax
                  kstep = 2
                END IF
              END IF
              kk = k + kstep - 1
              IF (kp/=kk) THEN
                a(kp, kp) = a(kk, kk)
                CALL DCOPY(kp-kk-1, a(kk+1,kk), 1_ip_, a(kp,kk+1), lda)
                IF (kp<n) CALL DCOPY(n-kp, a(kp+1,kk), 1_ip_, a(kp+1,kp),   &
                  1_ip_)
                IF (k>1) CALL DSWAP(k-1, a(kk,1), lda, a(kp,1), lda)
                CALL DSWAP(kk, w(kk,1), ldw, w(kp,1), ldw)
              END IF
              IF (kstep==1) THEN
                CALL DCOPY(n-k+1, w(k,k), 1_ip_, a(k,k), 1_ip_)
                IF (k<n) THEN
                  r1 = one/a(k, k)
                  CALL DSCAL(n-k, r1, a(k+1,k), 1_ip_)
                END IF
              ELSE
                IF (k<n-1) THEN
                  d21 = w(k+1, k)
                  d11 = w(k+1, k+1)/d21
                  d22 = w(k, k)/d21
                  t = one/(d11*d22-one)
                  d21 = t/d21
                  DO j = k + 2, n
                    a(j, k) = d21*(d11*w(j,k)-w(j,k+1))
                    a(j, k+1) = d21*(d22*w(j,k+1)-w(j,k))
                  END DO
                END IF
                a(k, k) = w(k, k)
                a(k+1, k) = w(k+1, k)
                a(k+1, k+1) = w(k+1, k+1)
              END IF
            END IF
            IF (kstep==1) THEN
              ipiv(k) = kp
            ELSE
              ipiv(k) = -kp
              ipiv(k+1) = -kp
            END IF
            k = k + kstep
            GO TO 70
 90         CONTINUE
            DO j = k, n, nb
              jb = MIN(nb, n-j+1)
              DO jj = j, j + jb - 1
                CALL DGEMV('No transpose', j+jb-jj, k-1, -one, a(jj,1),     &
                  lda, w(jj,1), ldw, one, a(jj,jj), 1_ip_)
              END DO
              IF (j+jb<=n) CALL DGEMM('No transpose', 'Transpose',          &
                n-j-jb+1, jb, k-1, -one, a(j+jb,1), lda, w(j,1), ldw, one,  &
                a(j+jb,j), lda)
            END DO
            j = k - 1
 120        CONTINUE
            jj = j
            jp = ipiv(j)
            IF (jp<0) THEN
              jp = -jp
              j = j - 1
            END IF
            j = j - 1
            IF (jp/=jj .AND. j>=1) CALL DSWAP(j, a(jp,1), lda, a(jj,1),     &
              lda)
            IF (j>1) GO TO 120
            kb = k - 1
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLATRD(uplo, n, nb, a, lda, e, tau, w, ldw)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: lda, ldw, n, nb
          REAL(rp_) :: a(lda, *), e(*), tau(*), w(ldw, *)
          REAL(rp_) :: zero, one, half
          PARAMETER (zero=0.0_rp_, one=1.0_rp_, half=0.5_rp_)
          INTEGER(ip_) :: i, iw
          REAL(rp_) :: alpha
          EXTERNAL :: DAXPY, DGEMV, DLARFG, DSCAL, DSYMV
          LOGICAL :: LSAME
          REAL(rp_) :: DDOT
          EXTERNAL :: LSAME, DDOT
          INTRINSIC :: MIN
          IF (n<=0) RETURN
          IF (LSAME(uplo,'U')) THEN
            DO i = n, n - nb + 1, -1_ip_
              iw = i - n + nb
              IF (i<n) THEN
                CALL DGEMV('No transpose', i, n-i, -one, a(1,i+1), lda,     &
                  w(i,iw+1), ldw, one, a(1,i), 1_ip_)
                CALL DGEMV('No transpose', i, n-i, -one, w(1,iw+1), ldw,    &
                  a(i,i+1), lda, one, a(1,i), 1_ip_)
              END IF
              IF (i>1) THEN
                CALL DLARFG(i-1, a(i-1,i), a(1,i), 1_ip_, tau(i-1))
                e(i-1) = a(i-1, i)
                a(i-1, i) = one
                CALL DSYMV('Upper', i-1, one, a, lda, a(1,i), 1_ip_, zero,  &
                  w(1,iw), 1_ip_)
                IF (i<n) THEN
                  CALL DGEMV('Transpose', i-1, n-i, one, w(1,iw+1), ldw,    &
                    a(1,i), 1_ip_, zero, w(i+1,iw), 1_ip_)
                  CALL DGEMV('No transpose', i-1, n-i, -one, a(1,i+1), lda, &
                    w(i+1,iw), 1_ip_, one, w(1,iw), 1_ip_)
                  CALL DGEMV('Transpose', i-1, n-i, one, a(1,i+1), lda,     &
                    a(1,i), 1_ip_, zero, w(i+1,iw), 1_ip_)
                  CALL DGEMV('No transpose', i-1, n-i, -one, w(1,iw+1),     &
                    ldw, w(i+1,iw), 1_ip_, one, w(1,iw), 1_ip_)
                END IF
                CALL DSCAL(i-1, tau(i-1), w(1,iw), 1_ip_)
                alpha = -half*tau(i-1)*DDOT(i-1, w(1,iw), 1_ip_, a(1,i),    &
                  1_ip_)
                CALL DAXPY(i-1, alpha, a(1,i), 1_ip_, w(1,iw), 1_ip_)
              END IF
            END DO
          ELSE
            DO i = 1, nb
              CALL DGEMV('No transpose', n-i+1, i-1, -one, a(i,1), lda,     &
                w(i,1), ldw, one, a(i,i), 1_ip_)
              CALL DGEMV('No transpose', n-i+1, i-1, -one, w(i,1), ldw,     &
                a(i,1), lda, one, a(i,i), 1_ip_)
              IF (i<n) THEN
                CALL DLARFG(n-i, a(i+1,i), a(MIN(i+2,n),i), 1_ip_, tau(i))
                e(i) = a(i+1, i)
                a(i+1, i) = one
                CALL DSYMV('Lower', n-i, one, a(i+1,i+1), lda, a(i+1,i),    &
                  1_ip_, zero, w(i+1,i), 1_ip_)
                CALL DGEMV('Transpose', n-i, i-1, one, w(i+1,1), ldw,       &
                  a(i+1,i), 1_ip_, zero, w(1,i), 1_ip_)
                CALL DGEMV('No transpose', n-i, i-1, -one, a(i+1,1), lda,   &
                  w(1,i), 1_ip_, one, w(i+1,i), 1_ip_)
                CALL DGEMV('Transpose', n-i, i-1, one, a(i+1,1), lda,       &
                  a(i+1,i), 1_ip_, zero, w(1,i), 1_ip_)
                CALL DGEMV('No transpose', n-i, i-1, -one, w(i+1,1), ldw,   &
                  w(1,i), 1_ip_, one, w(i+1,i), 1_ip_)
                CALL DSCAL(n-i, tau(i), w(i+1,i), 1_ip_)
                alpha = -half*tau(i)*DDOT(n-i, w(i+1,i), 1_ip_, a(i+1,i),   &
                  1_ip_)
                CALL DAXPY(n-i, alpha, a(i+1,i), 1_ip_, w(i+1,i), 1_ip_)
              END IF
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DLATRZ(m, n, l, a, lda, tau, work)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: l, lda, m, n
          REAL(rp_) :: a(lda, *), tau(*), work(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          INTEGER(ip_) :: i
          EXTERNAL :: DLARFG, DLARZ
          IF (m==0) THEN
            RETURN
          ELSE IF (m==n) THEN
            DO i = 1, n
              tau(i) = zero
            END DO
            RETURN
          END IF
          DO i = m, 1_ip_, -1_ip_
            CALL DLARFG(l+1, a(i,i), a(i,n-l+1), lda, tau(i))
            CALL DLARZ('Right', i-1, n-i+1, l, a(i,n-l+1), lda, tau(i),     &
              a(1,i), lda, work)
          END DO
          RETURN
        END SUBROUTINE

        SUBROUTINE DORG2L(m, n, k, a, lda, tau, work, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, k, lda, m, n
          REAL(rp_) :: a(lda, *), tau(*), work(*)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          INTEGER(ip_) :: i, ii, j, l
          EXTERNAL :: DLARF, DSCAL, XERBLA2
          INTRINSIC :: MAX
          info = 0
          IF (m<0) THEN
            info = -1
          ELSE IF (n<0 .OR. n>m) THEN
            info = -2
          ELSE IF (k<0 .OR. k>n) THEN
            info = -3
          ELSE IF (lda<MAX(1,m)) THEN
            info = -5
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('ORG2L', -info)
            RETURN
          END IF
          IF (n<=0) RETURN
          DO j = 1, n - k
            DO l = 1, m
              a(l, j) = zero
            END DO
            a(m-n+j, j) = one
          END DO
          DO i = 1, k
            ii = n - k + i
            a(m-n+ii, ii) = one
            CALL DLARF('Left', m-n+ii, ii-1, a(1,ii), 1_ip_, tau(i), a,     &
              lda, work)
            CALL DSCAL(m-n+ii-1, -tau(i), a(1,ii), 1_ip_)
            a(m-n+ii, ii) = one - tau(i)
            DO l = m - n + ii + 1, m
              a(l, ii) = zero
            END DO
          END DO
          RETURN
        END SUBROUTINE

        SUBROUTINE DORG2R(m, n, k, a, lda, tau, work, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, k, lda, m, n
          REAL(rp_) :: a(lda, *), tau(*), work(*)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          INTEGER(ip_) :: i, j, l
          EXTERNAL :: DLARF, DSCAL, XERBLA2
          INTRINSIC :: MAX
          info = 0
          IF (m<0) THEN
            info = -1
          ELSE IF (n<0 .OR. n>m) THEN
            info = -2
          ELSE IF (k<0 .OR. k>n) THEN
            info = -3
          ELSE IF (lda<MAX(1,m)) THEN
            info = -5
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('ORG2R', -info)
            RETURN
          END IF
          IF (n<=0) RETURN
          DO j = k + 1, n
            DO l = 1, m
              a(l, j) = zero
            END DO
            a(j, j) = one
          END DO
          DO i = k, 1_ip_, -1_ip_
            IF (i<n) THEN
              a(i, i) = one
              CALL DLARF('Left', m-i+1, n-i, a(i,i), 1_ip_, tau(i), a(i,    &
                i+1), lda, work)
            END IF
            IF (i<m) CALL DSCAL(m-i, -tau(i), a(i+1,i), 1_ip_)
            a(i, i) = one - tau(i)
            DO l = 1, i - 1
              a(l, i) = zero
            END DO
          END DO
          RETURN
        END SUBROUTINE

        SUBROUTINE DORGBR(vect, m, n, k, a, lda, tau, work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: vect
          INTEGER(ip_) :: info, k, lda, lwork, m, n
          REAL(rp_) :: a(lda, *), tau(*), work(*)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          LOGICAL :: lquery, wantq
          INTEGER(ip_) :: i, iinfo, j, lwkopt, mn
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: DORGLQ, DORGQR, XERBLA2
          INTRINSIC :: MAX, MIN
          info = 0
          wantq = LSAME(vect, 'Q')
          mn = MIN(m, n)
          lquery = (lwork==-1)
          IF (.NOT. wantq .AND. .NOT. LSAME(vect,'P')) THEN
            info = -1
          ELSE IF (m<0) THEN
            info = -2
          ELSE IF (n<0 .OR. (wantq .AND. (n>m .OR. n<MIN(m,k))) .OR. (      &
            .NOT. wantq .AND. (m>n .OR. m<MIN(n,k)))) THEN
            info = -3
          ELSE IF (k<0) THEN
            info = -4
          ELSE IF (lda<MAX(1,m)) THEN
            info = -6
          ELSE IF (lwork<MAX(1,mn) .AND. .NOT. lquery) THEN
            info = -9
          END IF
          IF (info==0) THEN
            work(1) = 1
            IF (wantq) THEN
              IF (m>=k) THEN
                CALL DORGQR(m, n, k, a, lda, tau, work, -1_ip_, iinfo)
              ELSE
                IF (m>1) THEN
                  CALL DORGQR(m-1, m-1, m-1, a(2,2), lda, tau, work,        &
                    -1_ip_, iinfo)
                END IF
              END IF
            ELSE
              IF (k<n) THEN
                CALL DORGLQ(m, n, k, a, lda, tau, work, -1_ip_, iinfo)
              ELSE
                IF (n>1) THEN
                  CALL DORGLQ(n-1, n-1, n-1, a(2,2), lda, tau, work,        &
                    -1_ip_, iinfo)
                END IF
              END IF
            END IF
            lwkopt = INT(work(1),ip_)
            lwkopt = MAX(lwkopt, mn)
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('ORGBR', -info)
            RETURN
          ELSE IF (lquery) THEN
            work(1) = REAL(lwkopt,rp_)
            RETURN
          END IF
          IF (m==0 .OR. n==0) THEN
            work(1) = 1
            RETURN
          END IF
          IF (wantq) THEN
            IF (m>=k) THEN
              CALL DORGQR(m, n, k, a, lda, tau, work, lwork, iinfo)
            ELSE
              DO j = m, 2_ip_, -1_ip_
                a(1, j) = zero
                DO i = j + 1, m
                  a(i, j) = a(i, j-1)
                END DO
              END DO
              a(1, 1_ip_) = one
              DO i = 2, m
                a(i, 1_ip_) = zero
              END DO
              IF (m>1) THEN
                CALL DORGQR(m-1, m-1, m-1, a(2,2), lda, tau, work, lwork,   &
                  iinfo)
              END IF
            END IF
          ELSE
            IF (k<n) THEN
              CALL DORGLQ(m, n, k, a, lda, tau, work, lwork, iinfo)
            ELSE
              a(1, 1_ip_) = one
              DO i = 2, n
                a(i, 1_ip_) = zero
              END DO
              DO j = 2, n
                DO i = j - 1, 2_ip_, -1_ip_
                  a(i, j) = a(i-1, j)
                END DO
                a(1, j) = zero
              END DO
              IF (n>1) THEN
                CALL DORGLQ(n-1, n-1, n-1, a(2,2), lda, tau, work, lwork,   &
                  iinfo)
              END IF
            END IF
          END IF
          work(1) = REAL(lwkopt,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DORGL2(m, n, k, a, lda, tau, work, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, k, lda, m, n
          REAL(rp_) :: a(lda, *), tau(*), work(*)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          INTEGER(ip_) :: i, j, l
          EXTERNAL :: DLARF, DSCAL, XERBLA2
          INTRINSIC :: MAX
          info = 0
          IF (m<0) THEN
            info = -1
          ELSE IF (n<m) THEN
            info = -2
          ELSE IF (k<0 .OR. k>m) THEN
            info = -3
          ELSE IF (lda<MAX(1,m)) THEN
            info = -5
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('ORGL2', -info)
            RETURN
          END IF
          IF (m<=0) RETURN
          IF (k<m) THEN
            DO j = 1, n
              DO l = k + 1, m
                a(l, j) = zero
              END DO
              IF (j>k .AND. j<=m) a(j, j) = one
            END DO
          END IF
          DO i = k, 1_ip_, -1_ip_
            IF (i<n) THEN
              IF (i<m) THEN
                a(i, i) = one
                CALL DLARF('Right', m-i, n-i+1, a(i,i), lda, tau(i), a(i+1, &
                  i), lda, work)
              END IF
              CALL DSCAL(n-i, -tau(i), a(i,i+1), lda)
            END IF
            a(i, i) = one - tau(i)
            DO l = 1, i - 1
              a(i, l) = zero
            END DO
          END DO
          RETURN
        END SUBROUTINE

        SUBROUTINE DORGLQ(m, n, k, a, lda, tau, work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, k, lda, lwork, m, n
          REAL(rp_) :: a(lda, *), tau(*), work(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          LOGICAL :: lquery
          INTEGER(ip_) :: i, ib, iinfo, iws, j, ki, kk, l, ldwork, lwkopt,  &
            nb, nbmin, nx
          EXTERNAL :: DLARFB, DLARFT, DORGL2, XERBLA2
          INTRINSIC :: MAX, MIN
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: ILAENV2
          info = 0
          nb = ILAENV2(1_ip_, 'ORGLQ', ' ', m, n, k, -1_ip_)
          lwkopt = MAX(1, m)*nb
          work(1) = REAL(lwkopt,rp_)
          lquery = (lwork==-1)
          IF (m<0) THEN
            info = -1
          ELSE IF (n<m) THEN
            info = -2
          ELSE IF (k<0 .OR. k>m) THEN
            info = -3
          ELSE IF (lda<MAX(1,m)) THEN
            info = -5
          ELSE IF (lwork<MAX(1,m) .AND. .NOT. lquery) THEN
            info = -8
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('ORGLQ', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          IF (m<=0) THEN
            work(1) = 1
            RETURN
          END IF
          nbmin = 2
          nx = 0
          iws = m
          IF (nb>1 .AND. nb<k) THEN
            nx = MAX(0_ip_, ILAENV2(3_ip_,'ORGLQ',' ',m,n,k,-1_ip_))
            IF (nx<k) THEN
              ldwork = m
              iws = ldwork*nb
              IF (lwork<iws) THEN
                nb = lwork/ldwork
                nbmin = MAX(2_ip_, ILAENV2(2_ip_,'ORGLQ',' ',m,n,k,-1_ip_))
              END IF
            END IF
          END IF
          IF (nb>=nbmin .AND. nb<k .AND. nx<k) THEN
            ki = ((k-nx-1)/nb)*nb
            kk = MIN(k, ki+nb)
            DO j = 1, kk
              DO i = kk + 1, m
                a(i, j) = zero
              END DO
            END DO
          ELSE
            kk = 0
          END IF
          IF (kk<m) CALL DORGL2(m-kk, n-kk, k-kk, a(kk+1,kk+1), lda,        &
            tau(kk+1), work, iinfo)
          IF (kk>0) THEN
            DO i = ki + 1, 1_ip_, -nb
              ib = MIN(nb, k-i+1)
              IF (i+ib<=m) THEN
                CALL DLARFT('Forward', 'Rowwise', n-i+1, ib, a(i,i), lda,   &
                  tau(i), work, ldwork)
                CALL DLARFB('Right', 'Transpose', 'Forward', 'Rowwise',     &
                  m-i-ib+1, n-i+1, ib, a(i,i), lda, work, ldwork, a(i+ib,i),&
                  lda, work(ib+1), ldwork)
              END IF
              CALL DORGL2(ib, n-i+1, ib, a(i,i), lda, tau(i), work, iinfo)
              DO j = 1, i - 1
                DO l = i, i + ib - 1
                  a(l, j) = zero
                END DO
              END DO
            END DO
          END IF
          work(1) = REAL(iws,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DORGQL(m, n, k, a, lda, tau, work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, k, lda, lwork, m, n
          REAL(rp_) :: a(lda, *), tau(*), work(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          LOGICAL :: lquery
          INTEGER(ip_) :: i, ib, iinfo, iws, j, kk, l, ldwork, lwkopt, nb,  &
            nbmin, nx
          EXTERNAL :: DLARFB, DLARFT, DORG2L, XERBLA2
          INTRINSIC :: MAX, MIN
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: ILAENV2
          info = 0
          lquery = (lwork==-1)
          IF (m<0) THEN
            info = -1
          ELSE IF (n<0 .OR. n>m) THEN
            info = -2
          ELSE IF (k<0 .OR. k>n) THEN
            info = -3
          ELSE IF (lda<MAX(1,m)) THEN
            info = -5
          END IF
          IF (info==0) THEN
            IF (n==0) THEN
              lwkopt = 1
            ELSE
              nb = ILAENV2(1_ip_, 'ORGQL', ' ', m, n, k, -1_ip_)
              lwkopt = n*nb
            END IF
            work(1) = REAL(lwkopt,rp_)
            IF (lwork<MAX(1,n) .AND. .NOT. lquery) THEN
              info = -8
            END IF
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('ORGQL', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          IF (n<=0) THEN
            RETURN
          END IF
          nbmin = 2
          nx = 0
          iws = n
          IF (nb>1 .AND. nb<k) THEN
            nx = MAX(0_ip_, ILAENV2(3_ip_,'ORGQL',' ',m,n,k,-1_ip_))
            IF (nx<k) THEN
              ldwork = n
              iws = ldwork*nb
              IF (lwork<iws) THEN
                nb = lwork/ldwork
                nbmin = MAX(2_ip_, ILAENV2(2_ip_,'ORGQL',' ',m,n,k,-1_ip_))
              END IF
            END IF
          END IF
          IF (nb>=nbmin .AND. nb<k .AND. nx<k) THEN
            kk = MIN(k, ((k-nx+nb-1)/nb)*nb)
            DO j = 1, n - kk
              DO i = m - kk + 1, m
                a(i, j) = zero
              END DO
            END DO
          ELSE
            kk = 0
          END IF
          CALL DORG2L(m-kk, n-kk, k-kk, a, lda, tau, work, iinfo)
          IF (kk>0) THEN
            DO i = k - kk + 1, k, nb
              ib = MIN(nb, k-i+1)
              IF (n-k+i>1) THEN
                CALL DLARFT('Backward', 'Columnwise', m-k+i+ib-1, ib, a(1,  &
                  n-k+i), lda, tau(i), work, ldwork)
                CALL DLARFB('Left', 'No transpose', 'Backward',             &
                  'Columnwise', m-k+i+ib-1, n-k+i-1, ib, a(1,n-k+i), lda,   &
                  work, ldwork, a, lda, work(ib+1), ldwork)
              END IF
              CALL DORG2L(m-k+i+ib-1, ib, ib, a(1,n-k+i), lda, tau(i),      &
                work, iinfo)
              DO j = n - k + i, n - k + i + ib - 1
                DO l = m - k + i + ib, m
                  a(l, j) = zero
                END DO
              END DO
            END DO
          END IF
          work(1) = REAL(iws,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DORGQR(m, n, k, a, lda, tau, work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, k, lda, lwork, m, n
          REAL(rp_) :: a(lda, *), tau(*), work(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          LOGICAL :: lquery
          INTEGER(ip_) :: i, ib, iinfo, iws, j, ki, kk, l, ldwork, lwkopt,  &
            nb, nbmin, nx
          EXTERNAL :: DLARFB, DLARFT, DORG2R, XERBLA2
          INTRINSIC :: MAX, MIN
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: ILAENV2
          info = 0
          nb = ILAENV2(1_ip_, 'ORGQR', ' ', m, n, k, -1_ip_)
          lwkopt = MAX(1, n)*nb
          work(1) = REAL(lwkopt,rp_)
          lquery = (lwork==-1)
          IF (m<0) THEN
            info = -1
          ELSE IF (n<0 .OR. n>m) THEN
            info = -2
          ELSE IF (k<0 .OR. k>n) THEN
            info = -3
          ELSE IF (lda<MAX(1,m)) THEN
            info = -5
          ELSE IF (lwork<MAX(1,n) .AND. .NOT. lquery) THEN
            info = -8
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('ORGQR', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          IF (n<=0) THEN
            work(1) = 1
            RETURN
          END IF
          nbmin = 2
          nx = 0
          iws = n
          IF (nb>1 .AND. nb<k) THEN
            nx = MAX(0_ip_, ILAENV2(3_ip_,'ORGQR',' ',m,n,k,-1_ip_))
            IF (nx<k) THEN
              ldwork = n
              iws = ldwork*nb
              IF (lwork<iws) THEN
                nb = lwork/ldwork
                nbmin = MAX(2_ip_, ILAENV2(2_ip_,'ORGQR',' ',m,n,k,-1_ip_))
              END IF
            END IF
          END IF
          IF (nb>=nbmin .AND. nb<k .AND. nx<k) THEN
            ki = ((k-nx-1)/nb)*nb
            kk = MIN(k, ki+nb)
            DO j = kk + 1, n
              DO i = 1, kk
                a(i, j) = zero
              END DO
            END DO
          ELSE
            kk = 0
          END IF
          IF (kk<n) CALL DORG2R(m-kk, n-kk, k-kk, a(kk+1,kk+1), lda,        &
            tau(kk+1), work, iinfo)
          IF (kk>0) THEN
            DO i = ki + 1, 1_ip_, -nb
              ib = MIN(nb, k-i+1)
              IF (i+ib<=n) THEN
                CALL DLARFT('Forward', 'Columnwise', m-i+1, ib, a(i,i),     &
                  lda, tau(i), work, ldwork)
                CALL DLARFB('Left', 'No transpose', 'Forward',              &
                  'Columnwise', m-i+1, n-i-ib+1, ib, a(i,i), lda, work,     &
                  ldwork, a(i,i+ib), lda, work(ib+1), ldwork)
              END IF
              CALL DORG2R(m-i+1, ib, ib, a(i,i), lda, tau(i), work, iinfo)
              DO j = i, i + ib - 1
                DO l = 1, i - 1
                  a(l, j) = zero
                END DO
              END DO
            END DO
          END IF
          work(1) = REAL(iws,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DORGTR(uplo, n, a, lda, tau, work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, lda, lwork, n
          REAL(rp_) :: a(lda, *), tau(*), work(*)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          LOGICAL :: lquery, upper
          INTEGER(ip_) :: i, iinfo, j, lwkopt, nb
          LOGICAL :: LSAME
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: LSAME, ILAENV2
          EXTERNAL :: DORGQL, DORGQR, XERBLA2
          INTRINSIC :: MAX
          info = 0
          lquery = (lwork==-1)
          upper = LSAME(uplo, 'U')
          IF (.NOT. upper .AND. .NOT. LSAME(uplo,'L')) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,n)) THEN
            info = -4
          ELSE IF (lwork<MAX(1,n-1) .AND. .NOT. lquery) THEN
            info = -7
          END IF
          IF (info==0) THEN
            IF (upper) THEN
              nb = ILAENV2(1_ip_, 'ORGQL', ' ', n-1, n-1, n-1, -1_ip_)
            ELSE
              nb = ILAENV2(1_ip_, 'ORGQR', ' ', n-1, n-1, n-1, -1_ip_)
            END IF
            lwkopt = MAX(1, n-1)*nb
            work(1) = REAL(lwkopt,rp_)
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('ORGTR', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          IF (n==0) THEN
            work(1) = 1
            RETURN
          END IF
          IF (upper) THEN
            DO j = 1, n - 1
              DO i = 1, j - 1
                a(i, j) = a(i, j+1)
              END DO
              a(n, j) = zero
            END DO
            DO i = 1, n - 1
              a(i, n) = zero
            END DO
            a(n, n) = one
            CALL DORGQL(n-1, n-1, n-1, a, lda, tau, work, lwork, iinfo)
          ELSE
            DO j = n, 2_ip_, -1_ip_
              a(1, j) = zero
              DO i = j + 1, n
                a(i, j) = a(i, j-1)
              END DO
            END DO
            a(1, 1_ip_) = one
            DO i = 2, n
              a(i, 1_ip_) = zero
            END DO
            IF (n>1) THEN
              CALL DORGQR(n-1, n-1, n-1, a(2,2), lda, tau, work, lwork,     &
                iinfo)
            END IF
          END IF
          work(1) = REAL(lwkopt,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DORM2R(side, trans, m, n, k, a, lda, tau, c, ldc, work,  &
          info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: side, trans
          INTEGER(ip_) :: info, k, lda, ldc, m, n
          REAL(rp_) :: a(lda, *), c(ldc, *), tau(*), work(*)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          LOGICAL :: left, notran
          INTEGER(ip_) :: i, i1, i2, i3, ic, jc, mi, ni, nq
          REAL(rp_) :: aii
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: DLARF, XERBLA2
          INTRINSIC :: MAX
          info = 0
          left = LSAME(side, 'L')
          notran = LSAME(trans, 'N')
          IF (left) THEN
            nq = m
          ELSE
            nq = n
          END IF
          IF (.NOT. left .AND. .NOT. LSAME(side,'R')) THEN
            info = -1
          ELSE IF (.NOT. notran .AND. .NOT. LSAME(trans,'T')) THEN
            info = -2
          ELSE IF (m<0) THEN
            info = -3
          ELSE IF (n<0) THEN
            info = -4
          ELSE IF (k<0 .OR. k>nq) THEN
            info = -5
          ELSE IF (lda<MAX(1,nq)) THEN
            info = -7
          ELSE IF (ldc<MAX(1,m)) THEN
            info = -10
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('ORM2R', -info)
            RETURN
          END IF
          IF (m==0 .OR. n==0 .OR. k==0) RETURN
          IF ((left .AND. .NOT. notran) .OR. (.NOT. left .AND. notran))     &
            THEN
            i1 = 1
            i2 = k
            i3 = 1
          ELSE
            i1 = k
            i2 = 1
            i3 = -1
          END IF
          IF (left) THEN
            ni = n
            jc = 1
          ELSE
            mi = m
            ic = 1
          END IF
          DO i = i1, i2, i3
            IF (left) THEN
              mi = m - i + 1
              ic = i
            ELSE
              ni = n - i + 1
              jc = i
            END IF
            aii = a(i, i)
            a(i, i) = one
            CALL DLARF(side, mi, ni, a(i,i), 1_ip_, tau(i), c(ic,jc), ldc,  &
              work)
            a(i, i) = aii
          END DO
          RETURN
        END SUBROUTINE

        SUBROUTINE DORMBR(vect, side, trans, m, n, k, a, lda, tau, c, ldc,  &
          work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: side, trans, vect
          INTEGER(ip_) :: info, k, lda, ldc, lwork, m, n
          REAL(rp_) :: a(lda, *), c(ldc, *), tau(*), work(*)
          LOGICAL :: applyq, left, lquery, notran
          CHARACTER :: transt
          INTEGER(ip_) :: i1, i2, iinfo, lwkopt, mi, nb, ni, nq, nw
          LOGICAL :: LSAME
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: LSAME, ILAENV2
          EXTERNAL :: DORMLQ, DORMQR, XERBLA2
          INTRINSIC :: MAX, MIN
          info = 0
          applyq = LSAME(vect, 'Q')
          left = LSAME(side, 'L')
          notran = LSAME(trans, 'N')
          lquery = (lwork==-1)
          IF (left) THEN
            nq = m
            nw = n
          ELSE
            nq = n
            nw = m
          END IF
          IF (.NOT. applyq .AND. .NOT. LSAME(vect,'P')) THEN
            info = -1
          ELSE IF (.NOT. left .AND. .NOT. LSAME(side,'R')) THEN
            info = -2
          ELSE IF (.NOT. notran .AND. .NOT. LSAME(trans,'T')) THEN
            info = -3
          ELSE IF (m<0) THEN
            info = -4
          ELSE IF (n<0) THEN
            info = -5
          ELSE IF (k<0) THEN
            info = -6
          ELSE IF ((applyq .AND. lda<MAX(1,nq)) .OR. (.NOT. applyq .AND.    &
            lda<MAX(1,MIN(nq,k)))) THEN
            info = -8
          ELSE IF (ldc<MAX(1,m)) THEN
            info = -11
          ELSE IF (lwork<MAX(1,nw) .AND. .NOT. lquery) THEN
            info = -13
          END IF
          IF (info==0) THEN
            IF (applyq) THEN
              IF (left) THEN
                nb = ILAENV2(1_ip_, 'ORMQR', side//trans, m-1, n, m-1,      &
                  -1_ip_)
              ELSE
                nb = ILAENV2(1_ip_, 'ORMQR', side//trans, m, n-1, n-1,      &
                  -1_ip_)
              END IF
            ELSE
              IF (left) THEN
                nb = ILAENV2(1_ip_, 'ORMLQ', side//trans, m-1, n, m-1,      &
                  -1_ip_)
              ELSE
                nb = ILAENV2(1_ip_, 'ORMLQ', side//trans, m, n-1, n-1,      &
                  -1_ip_)
              END IF
            END IF
            lwkopt = MAX(1, nw)*nb
            work(1) = REAL(lwkopt,rp_)
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('ORMBR', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          work(1) = 1
          IF (m==0 .OR. n==0) RETURN
          IF (applyq) THEN
            IF (nq>=k) THEN
              CALL DORMQR(side, trans, m, n, k, a, lda, tau, c, ldc, work,  &
                lwork, iinfo)
            ELSE IF (nq>1) THEN
              IF (left) THEN
                mi = m - 1
                ni = n
                i1 = 2
                i2 = 1
              ELSE
                mi = m
                ni = n - 1
                i1 = 1
                i2 = 2
              END IF
              CALL DORMQR(side, trans, mi, ni, nq-1, a(2,1), lda, tau,      &
                c(i1,i2), ldc, work, lwork, iinfo)
            END IF
          ELSE
            IF (notran) THEN
              transt = 'T'
            ELSE
              transt = 'N'
            END IF
            IF (nq>k) THEN
              CALL DORMLQ(side, transt, m, n, k, a, lda, tau, c, ldc, work, &
                lwork, iinfo)
            ELSE IF (nq>1) THEN
              IF (left) THEN
                mi = m - 1
                ni = n
                i1 = 2
                i2 = 1
              ELSE
                mi = m
                ni = n - 1
                i1 = 1
                i2 = 2
              END IF
              CALL DORMLQ(side, transt, mi, ni, nq-1, a(1,2), lda, tau,     &
                c(i1,i2), ldc, work, lwork, iinfo)
            END IF
          END IF
          work(1) = REAL(lwkopt,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DORMHR(side, trans, m, n, ilo, ihi, a, lda, tau, c, ldc, &
          work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: side, trans
          INTEGER(ip_) :: ihi, ilo, info, lda, ldc, lwork, m, n
          REAL(rp_) :: a(lda, *), c(ldc, *), tau(*), work(*)
          LOGICAL :: left, lquery
          INTEGER(ip_) :: i1, i2, iinfo, lwkopt, mi, nb, nh, ni, nq, nw
          LOGICAL :: LSAME
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: LSAME, ILAENV2
          EXTERNAL :: DORMQR, XERBLA2
          INTRINSIC :: MAX, MIN
          info = 0
          nh = ihi - ilo
          left = LSAME(side, 'L')
          lquery = (lwork==-1)
          IF (left) THEN
            nq = m
            nw = n
          ELSE
            nq = n
            nw = m
          END IF
          IF (.NOT. left .AND. .NOT. LSAME(side,'R')) THEN
            info = -1
          ELSE IF (.NOT. LSAME(trans,'N') .AND. .NOT. LSAME(trans,& 
            'T')) THEN
            info = -2
          ELSE IF (m<0) THEN
            info = -3
          ELSE IF (n<0) THEN
            info = -4
          ELSE IF (ilo<1 .OR. ilo>MAX(1,nq)) THEN
            info = -5
          ELSE IF (ihi<MIN(ilo,nq) .OR. ihi>nq) THEN
            info = -6
          ELSE IF (lda<MAX(1,nq)) THEN
            info = -8
          ELSE IF (ldc<MAX(1,m)) THEN
            info = -11
          ELSE IF (lwork<MAX(1,nw) .AND. .NOT. lquery) THEN
            info = -13
          END IF
          IF (info==0) THEN
            IF (left) THEN
              nb = ILAENV2(1_ip_, 'ORMQR', side//trans, nh, n, nh, -1_ip_)
            ELSE
              nb = ILAENV2(1_ip_, 'ORMQR', side//trans, m, nh, nh, -1_ip_)
            END IF
            lwkopt = MAX(1, nw)*nb
            work(1) = REAL(lwkopt,rp_)
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('ORMHR', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          IF (m==0 .OR. n==0 .OR. nh==0) THEN
            work(1) = 1
            RETURN
          END IF
          IF (left) THEN
            mi = nh
            ni = n
            i1 = ilo + 1
            i2 = 1
          ELSE
            mi = m
            ni = nh
            i1 = 1
            i2 = ilo + 1
          END IF
          CALL DORMQR(side, trans, mi, ni, nh, a(ilo+1,ilo), lda, tau(ilo), &
            c(i1,i2), ldc, work, lwork, iinfo)
          work(1) = REAL(lwkopt,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DORML2(side, trans, m, n, k, a, lda, tau, c, ldc, work,  &
          info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: side, trans
          INTEGER(ip_) :: info, k, lda, ldc, m, n
          REAL(rp_) :: a(lda, *), c(ldc, *), tau(*), work(*)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          LOGICAL :: left, notran
          INTEGER(ip_) :: i, i1, i2, i3, ic, jc, mi, ni, nq
          REAL(rp_) :: aii
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: DLARF, XERBLA2
          INTRINSIC :: MAX
          info = 0
          left = LSAME(side, 'L')
          notran = LSAME(trans, 'N')
          IF (left) THEN
            nq = m
          ELSE
            nq = n
          END IF
          IF (.NOT. left .AND. .NOT. LSAME(side,'R')) THEN
            info = -1
          ELSE IF (.NOT. notran .AND. .NOT. LSAME(trans,'T')) THEN
            info = -2
          ELSE IF (m<0) THEN
            info = -3
          ELSE IF (n<0) THEN
            info = -4
          ELSE IF (k<0 .OR. k>nq) THEN
            info = -5
          ELSE IF (lda<MAX(1,k)) THEN
            info = -7
          ELSE IF (ldc<MAX(1,m)) THEN
            info = -10
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('ORML2', -info)
            RETURN
          END IF
          IF (m==0 .OR. n==0 .OR. k==0) RETURN
          IF ((left .AND. notran) .OR. (.NOT. left .AND. .NOT. notran))     &
            THEN
            i1 = 1
            i2 = k
            i3 = 1
          ELSE
            i1 = k
            i2 = 1
            i3 = -1
          END IF
          IF (left) THEN
            ni = n
            jc = 1
          ELSE
            mi = m
            ic = 1
          END IF
          DO i = i1, i2, i3
            IF (left) THEN
              mi = m - i + 1
              ic = i
            ELSE
              ni = n - i + 1
              jc = i
            END IF
            aii = a(i, i)
            a(i, i) = one
            CALL DLARF(side, mi, ni, a(i,i), lda, tau(i), c(ic,jc), ldc,    &
              work)
            a(i, i) = aii
          END DO
          RETURN
        END SUBROUTINE

        SUBROUTINE DORMLQ(side, trans, m, n, k, a, lda, tau, c, ldc, work,  &
          lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: side, trans
          INTEGER(ip_) :: info, k, lda, ldc, lwork, m, n
          REAL(rp_) :: a(lda, *), c(ldc, *), tau(*), work(*)
          INTEGER(ip_) :: nbmax, ldt, tsize
          PARAMETER (nbmax=64, ldt=nbmax+1, tsize=ldt*nbmax)
          LOGICAL :: left, lquery, notran
          CHARACTER :: transt
          INTEGER(ip_) :: i, i1, i2, i3, ib, ic, iinfo, iwt, jc, ldwork,    &
            lwkopt, mi, nb, nbmin, ni, nq, nw
          LOGICAL :: LSAME
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: LSAME, ILAENV2
          EXTERNAL :: DLARFB, DLARFT, DORML2, XERBLA2
          INTRINSIC :: MAX, MIN
          info = 0
          left = LSAME(side, 'L')
          notran = LSAME(trans, 'N')
          lquery = (lwork==-1)
          IF (left) THEN
            nq = m
            nw = n
          ELSE
            nq = n
            nw = m
          END IF
          IF (.NOT. left .AND. .NOT. LSAME(side,'R')) THEN
            info = -1
          ELSE IF (.NOT. notran .AND. .NOT. LSAME(trans,'T')) THEN
            info = -2
          ELSE IF (m<0) THEN
            info = -3
          ELSE IF (n<0) THEN
            info = -4
          ELSE IF (k<0 .OR. k>nq) THEN
            info = -5
          ELSE IF (lda<MAX(1,k)) THEN
            info = -7
          ELSE IF (ldc<MAX(1,m)) THEN
            info = -10
          ELSE IF (lwork<MAX(1,nw) .AND. .NOT. lquery) THEN
            info = -12
          END IF
          IF (info==0) THEN
            nb = MIN(nbmax, ILAENV2(1_ip_,'ORMLQ',side//trans,m,n,k,        &
              -1_ip_))
            lwkopt = MAX(1, nw)*nb + tsize
            work(1) = REAL(lwkopt,rp_)
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('ORMLQ', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          IF (m==0 .OR. n==0 .OR. k==0) THEN
            work(1) = 1
            RETURN
          END IF
          nbmin = 2
          ldwork = nw
          IF (nb>1 .AND. nb<k) THEN
            IF (lwork<nw*nb+tsize) THEN
              nb = (lwork-tsize)/ldwork
              nbmin = MAX(2_ip_, ILAENV2(2_ip_,'ORMLQ',side//trans,m,n,k,   &
                -1_ip_))
            END IF
          END IF
          IF (nb<nbmin .OR. nb>=k) THEN
            CALL DORML2(side, trans, m, n, k, a, lda, tau, c, ldc, work,    &
              iinfo)
          ELSE
            iwt = 1 + nw*nb
            IF ((left .AND. notran) .OR. (.NOT. left .AND. .NOT. notran))   &
              THEN
              i1 = 1
              i2 = k
              i3 = nb
            ELSE
              i1 = ((k-1)/nb)*nb + 1
              i2 = 1
              i3 = -nb
            END IF
            IF (left) THEN
              ni = n
              jc = 1
            ELSE
              mi = m
              ic = 1
            END IF
            IF (notran) THEN
              transt = 'T'
            ELSE
              transt = 'N'
            END IF
            DO i = i1, i2, i3
              ib = MIN(nb, k-i+1)
              CALL DLARFT('Forward', 'Rowwise', nq-i+1, ib, a(i,i), lda,    &
                tau(i), work(iwt), ldt)
              IF (left) THEN
                mi = m - i + 1
                ic = i
              ELSE
                ni = n - i + 1
                jc = i
              END IF
              CALL DLARFB(side, transt, 'Forward', 'Rowwise', mi, ni, ib,   &
                a(i,i), lda, work(iwt), ldt, c(ic,jc), ldc, work, ldwork)
            END DO
          END IF
          work(1) = REAL(lwkopt,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DORMQR(side, trans, m, n, k, a, lda, tau, c, ldc, work,  &
          lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: side, trans
          INTEGER(ip_) :: info, k, lda, ldc, lwork, m, n
          REAL(rp_) :: a(lda, *), c(ldc, *), tau(*), work(*)
          INTEGER(ip_) :: nbmax, ldt, tsize
          PARAMETER (nbmax=64, ldt=nbmax+1, tsize=ldt*nbmax)
          LOGICAL :: left, lquery, notran
          INTEGER(ip_) :: i, i1, i2, i3, ib, ic, iinfo, iwt, jc, ldwork,    &
            lwkopt, mi, nb, nbmin, ni, nq, nw
          LOGICAL :: LSAME
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: LSAME, ILAENV2
          EXTERNAL :: DLARFB, DLARFT, DORM2R, XERBLA2
          INTRINSIC :: MAX, MIN
          info = 0
          left = LSAME(side, 'L')
          notran = LSAME(trans, 'N')
          lquery = (lwork==-1)
          IF (left) THEN
            nq = m
            nw = n
          ELSE
            nq = n
            nw = m
          END IF
          IF (.NOT. left .AND. .NOT. LSAME(side,'R')) THEN
            info = -1
          ELSE IF (.NOT. notran .AND. .NOT. LSAME(trans,'T')) THEN
            info = -2
          ELSE IF (m<0) THEN
            info = -3
          ELSE IF (n<0) THEN
            info = -4
          ELSE IF (k<0 .OR. k>nq) THEN
            info = -5
          ELSE IF (lda<MAX(1,nq)) THEN
            info = -7
          ELSE IF (ldc<MAX(1,m)) THEN
            info = -10
          ELSE IF (lwork<MAX(1,nw) .AND. .NOT. lquery) THEN
            info = -12
          END IF
          IF (info==0) THEN
            nb = MIN(nbmax, ILAENV2(1_ip_,'ORMQR',side//trans,m,n,k,        &
              -1_ip_))
            lwkopt = MAX(1, nw)*nb + tsize
            work(1) = REAL(lwkopt,rp_)
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('ORMQR', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          IF (m==0 .OR. n==0 .OR. k==0) THEN
            work(1) = 1
            RETURN
          END IF
          nbmin = 2
          ldwork = nw
          IF (nb>1 .AND. nb<k) THEN
            IF (lwork<nw*nb+tsize) THEN
              nb = (lwork-tsize)/ldwork
              nbmin = MAX(2_ip_, ILAENV2(2_ip_,'ORMQR',side//trans,m,n,k,   &
                -1_ip_))
            END IF
          END IF
          IF (nb<nbmin .OR. nb>=k) THEN
            CALL DORM2R(side, trans, m, n, k, a, lda, tau, c, ldc, work,    &
              iinfo)
          ELSE
            iwt = 1 + nw*nb
            IF ((left .AND. .NOT. notran) .OR. (.NOT. left .AND. notran))   &
              THEN
              i1 = 1
              i2 = k
              i3 = nb
            ELSE
              i1 = ((k-1)/nb)*nb + 1
              i2 = 1
              i3 = -nb
            END IF
            IF (left) THEN
              ni = n
              jc = 1
            ELSE
              mi = m
              ic = 1
            END IF
            DO i = i1, i2, i3
              ib = MIN(nb, k-i+1)
              CALL DLARFT('Forward', 'Columnwise', nq-i+1, ib, a(i,i), lda, &
                tau(i), work(iwt), ldt)
              IF (left) THEN
                mi = m - i + 1
                ic = i
              ELSE
                ni = n - i + 1
                jc = i
              END IF
              CALL DLARFB(side, trans, 'Forward', 'Columnwise', mi, ni, ib, &
                a(i,i), lda, work(iwt), ldt, c(ic,jc), ldc, work, ldwork)
            END DO
          END IF
          work(1) = REAL(lwkopt,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DORMR3(side, trans, m, n, k, l, a, lda, tau, c, ldc,     &
          work, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: side, trans
          INTEGER(ip_) :: info, k, l, lda, ldc, m, n
          REAL(rp_) :: a(lda, *), c(ldc, *), tau(*), work(*)
          LOGICAL :: left, notran
          INTEGER(ip_) :: i, i1, i2, i3, ic, ja, jc, mi, ni, nq
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: DLARZ, XERBLA2
          INTRINSIC :: MAX
          info = 0
          left = LSAME(side, 'L')
          notran = LSAME(trans, 'N')
          IF (left) THEN
            nq = m
          ELSE
            nq = n
          END IF
          IF (.NOT. left .AND. .NOT. LSAME(side,'R')) THEN
            info = -1
          ELSE IF (.NOT. notran .AND. .NOT. LSAME(trans,'T')) THEN
            info = -2
          ELSE IF (m<0) THEN
            info = -3
          ELSE IF (n<0) THEN
            info = -4
          ELSE IF (k<0 .OR. k>nq) THEN
            info = -5
          ELSE IF (l<0 .OR. (left .AND. (l>m)) .OR. (.NOT. left .AND. (l>   &
            n))) THEN
            info = -6
          ELSE IF (lda<MAX(1,k)) THEN
            info = -8
          ELSE IF (ldc<MAX(1,m)) THEN
            info = -11
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('ORMR3', -info)
            RETURN
          END IF
          IF (m==0 .OR. n==0 .OR. k==0) RETURN
          IF ((left .AND. .NOT. notran .OR. .NOT. left .AND. notran)) THEN
            i1 = 1
            i2 = k
            i3 = 1
          ELSE
            i1 = k
            i2 = 1
            i3 = -1
          END IF
          IF (left) THEN
            ni = n
            ja = m - l + 1
            jc = 1
          ELSE
            mi = m
            ja = n - l + 1
            ic = 1
          END IF
          DO i = i1, i2, i3
            IF (left) THEN
              mi = m - i + 1
              ic = i
            ELSE
              ni = n - i + 1
              jc = i
            END IF
            CALL DLARZ(side, mi, ni, l, a(i,ja), lda, tau(i), c(ic,jc),     &
              ldc, work)
          END DO
          RETURN
        END SUBROUTINE

        SUBROUTINE DORMRZ(side, trans, m, n, k, l, a, lda, tau, c, ldc,     &
          work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: side, trans
          INTEGER(ip_) :: info, k, l, lda, ldc, lwork, m, n
          REAL(rp_) :: a(lda, *), c(ldc, *), tau(*), work(*)
          INTEGER(ip_) :: nbmax, ldt, tsize
          PARAMETER (nbmax=64, ldt=nbmax+1, tsize=ldt*nbmax)
          LOGICAL :: left, lquery, notran
          CHARACTER :: transt
          INTEGER(ip_) :: i, i1, i2, i3, ib, ic, iinfo, iwt, ja, jc,        &
            ldwork, lwkopt, mi, nb, nbmin, ni, nq, nw
          LOGICAL :: LSAME
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: LSAME, ILAENV2
          EXTERNAL :: DLARZB, DLARZT, DORMR3, XERBLA2
          INTRINSIC :: MAX, MIN
          info = 0
          left = LSAME(side, 'L')
          notran = LSAME(trans, 'N')
          lquery = (lwork==-1)
          IF (left) THEN
            nq = m
            nw = MAX(1, n)
          ELSE
            nq = n
            nw = MAX(1, m)
          END IF
          IF (.NOT. left .AND. .NOT. LSAME(side,'R')) THEN
            info = -1
          ELSE IF (.NOT. notran .AND. .NOT. LSAME(trans,'T')) THEN
            info = -2
          ELSE IF (m<0) THEN
            info = -3
          ELSE IF (n<0) THEN
            info = -4
          ELSE IF (k<0 .OR. k>nq) THEN
            info = -5
          ELSE IF (l<0 .OR. (left .AND. (l>m)) .OR. (.NOT. left .AND. (l>   &
            n))) THEN
            info = -6
          ELSE IF (lda<MAX(1,k)) THEN
            info = -8
          ELSE IF (ldc<MAX(1,m)) THEN
            info = -11
          ELSE IF (lwork<MAX(1,nw) .AND. .NOT. lquery) THEN
            info = -13
          END IF
          IF (info==0) THEN
            IF (m==0 .OR. n==0) THEN
              lwkopt = 1
            ELSE
              nb = MIN(nbmax, ILAENV2(1_ip_,'ORMRQ',side//trans,m,n,k,      &
                -1_ip_))
              lwkopt = nw*nb + tsize
            END IF
            work(1) = REAL(lwkopt,rp_)
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('ORMRZ', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          IF (m==0 .OR. n==0) THEN
            work(1) = 1
            RETURN
          END IF
          nbmin = 2
          ldwork = nw
          IF (nb>1 .AND. nb<k) THEN
            IF (lwork<nw*nb+tsize) THEN
              nb = (lwork-tsize)/ldwork
              nbmin = MAX(2_ip_, ILAENV2(2_ip_,'ORMRQ',side//trans,m,n,k,   &
                -1_ip_))
            END IF
          END IF
          IF (nb<nbmin .OR. nb>=k) THEN
            CALL DORMR3(side, trans, m, n, k, l, a, lda, tau, c, ldc, work, &
              iinfo)
          ELSE
            iwt = 1 + nw*nb
            IF ((left .AND. .NOT. notran) .OR. (.NOT. left .AND. notran))   &
              THEN
              i1 = 1
              i2 = k
              i3 = nb
            ELSE
              i1 = ((k-1)/nb)*nb + 1
              i2 = 1
              i3 = -nb
            END IF
            IF (left) THEN
              ni = n
              jc = 1
              ja = m - l + 1
            ELSE
              mi = m
              ic = 1
              ja = n - l + 1
            END IF
            IF (notran) THEN
              transt = 'T'
            ELSE
              transt = 'N'
            END IF
            DO i = i1, i2, i3
              ib = MIN(nb, k-i+1)
              CALL DLARZT('Backward', 'Rowwise', l, ib, a(i,ja), lda,       &
                tau(i), work(iwt), ldt)
              IF (left) THEN
                mi = m - i + 1
                ic = i
              ELSE
                ni = n - i + 1
                jc = i
              END IF
              CALL DLARZB(side, transt, 'Backward', 'Rowwise', mi, ni, ib,  &
                l, a(i,ja), lda, work(iwt), ldt, c(ic,jc), ldc, work, ldwork)
            END DO
          END IF
          work(1) = REAL(lwkopt,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DPBTF2(uplo, n, kd, ab, ldab, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, kd, ldab, n
          REAL(rp_) :: ab(ldab, *)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          LOGICAL :: upper
          INTEGER(ip_) :: j, kld, kn
          REAL(rp_) :: ajj
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: DSCAL, DSYR, XERBLA2
          INTRINSIC :: MAX, MIN, SQRT
          info = 0
          upper = LSAME(uplo, 'U')
          IF (.NOT. upper .AND. .NOT. LSAME(uplo,'L')) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (kd<0) THEN
            info = -3
          ELSE IF (ldab<kd+1) THEN
            info = -5
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('PBTF2', -info)
            RETURN
          END IF
          IF (n==0) RETURN
          kld = MAX(1, ldab-1)
          IF (upper) THEN
            DO j = 1, n
              ajj = ab(kd+1, j)
              IF (ajj<=zero) GO TO 30
              ajj = SQRT(ajj)
              ab(kd+1, j) = ajj
              kn = MIN(kd, n-j)
              IF (kn>0) THEN
                CALL DSCAL(kn, one/ajj, ab(kd,j+1), kld)
                CALL DSYR('Upper', kn, -one, ab(kd,j+1), kld, ab(kd+1,j+1), &
                  kld)
              END IF
            END DO
          ELSE
            DO j = 1, n
              ajj = ab(1, j)
              IF (ajj<=zero) GO TO 30
              ajj = SQRT(ajj)
              ab(1, j) = ajj
              kn = MIN(kd, n-j)
              IF (kn>0) THEN
                CALL DSCAL(kn, one/ajj, ab(2,j), 1_ip_)
                CALL DSYR('Lower', kn, -one, ab(2,j), 1_ip_, ab(1,j+1),     &
                  kld)
              END IF
            END DO
          END IF
          RETURN
 30       CONTINUE
          info = j
          RETURN
        END SUBROUTINE

        SUBROUTINE DPBTRF(uplo, n, kd, ab, ldab, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, kd, ldab, n
          REAL(rp_) :: ab(ldab, *)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          INTEGER(ip_) :: nbmax, ldwork
          PARAMETER (nbmax=32, ldwork=nbmax+1)
          INTEGER(ip_) :: i, i2, i3, ib, ii, j, jj, nb
          REAL(rp_) :: work(ldwork, nbmax)
          LOGICAL :: LSAME
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: LSAME, ILAENV2
          EXTERNAL :: DGEMM, DPBTF2, DPOTF2, DSYRK, DTRSM, &
            XERBLA2
          INTRINSIC :: MIN
          info = 0
          IF ((.NOT. LSAME(uplo,'U')) .AND. (.NOT. LSAME(uplo,'L'))) &
            THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (kd<0) THEN
            info = -3
          ELSE IF (ldab<kd+1) THEN
            info = -5
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('PBTRF', -info)
            RETURN
          END IF
          IF (n==0) RETURN
          nb = ILAENV2(1_ip_, 'PBTRF', uplo, n, kd, -1_ip_, -1_ip_)
          nb = MIN(nb, nbmax)
          IF (nb<=1 .OR. nb>kd) THEN
            CALL DPBTF2(uplo, n, kd, ab, ldab, info)
          ELSE
            IF (LSAME(uplo,'U')) THEN
              DO j = 1, nb
                DO i = 1, j - 1
                  work(i, j) = zero
                END DO
              END DO
              DO i = 1, n, nb
                ib = MIN(nb, n-i+1)
                CALL DPOTF2(uplo, ib, ab(kd+1,i), ldab-1, ii)
                IF (ii/=0) THEN
                  info = i + ii - 1
                  GO TO 150
                END IF
                IF (i+ib<=n) THEN
                  i2 = MIN(kd-ib, n-i-ib+1)
                  i3 = MIN(ib, n-i-kd+1)
                  IF (i2>0) THEN
                    CALL DTRSM('Left', 'Upper', 'Transpose', 'Non-unit',    &
                      ib, i2, one, ab(kd+1,i), ldab-1, ab(kd+1-ib,i+ib),    &
                      ldab-1)
                    CALL DSYRK('Upper', 'Transpose', i2, ib, -one,          &
                      ab(kd+1-ib,i+ib), ldab-1, one, ab(kd+1,i+ib), ldab-1)
                  END IF
                  IF (i3>0) THEN
                    DO jj = 1, i3
                      DO ii = jj, ib
                        work(ii, jj) = ab(ii-jj+1, jj+i+kd-1)
                      END DO
                    END DO
                    CALL DTRSM('Left', 'Upper', 'Transpose', 'Non-unit',    &
                      ib, i3, one, ab(kd+1,i), ldab-1, work, ldwork)
                    IF (i2>0) CALL DGEMM('Transpose', 'No Transpose', i2,   &
                      i3, ib, -one, ab(kd+1-ib,i+ib), ldab-1, work, ldwork, &
                      one, ab(1+ib,i+kd), ldab-1)
                    CALL DSYRK('Upper', 'Transpose', i3, ib, -one, work,    &
                      ldwork, one, ab(kd+1,i+kd), ldab-1)
                    DO jj = 1, i3
                      DO ii = jj, ib
                        ab(ii-jj+1, jj+i+kd-1) = work(ii, jj)
                      END DO
                    END DO
                  END IF
                END IF
              END DO
            ELSE
              DO j = 1, nb
                DO i = j + 1, nb
                  work(i, j) = zero
                END DO
              END DO
              DO i = 1, n, nb
                ib = MIN(nb, n-i+1)
                CALL DPOTF2(uplo, ib, ab(1,i), ldab-1, ii)
                IF (ii/=0) THEN
                  info = i + ii - 1
                  GO TO 150
                END IF
                IF (i+ib<=n) THEN
                  i2 = MIN(kd-ib, n-i-ib+1)
                  i3 = MIN(ib, n-i-kd+1)
                  IF (i2>0) THEN
                    CALL DTRSM('Right', 'Lower', 'Transpose', 'Non-unit',   &
                      i2, ib, one, ab(1,i), ldab-1, ab(1+ib,i), ldab-1)
                    CALL DSYRK('Lower', 'No Transpose', i2, ib, -one,       &
                      ab(1+ib,i), ldab-1, one, ab(1,i+ib), ldab-1)
                  END IF
                  IF (i3>0) THEN
                    DO jj = 1, ib
                      DO ii = 1, MIN(jj, i3)
                        work(ii, jj) = ab(kd+1-jj+ii, jj+i-1)
                      END DO
                    END DO
                    CALL DTRSM('Right', 'Lower', 'Transpose', 'Non-unit',   &
                      i3, ib, one, ab(1,i), ldab-1, work, ldwork)
                    IF (i2>0) CALL DGEMM('No transpose', 'Transpose', i3,   &
                      i2, ib, -one, work, ldwork, ab(1+ib,i), ldab-1, one,  &
                      ab(1+kd-ib,i+ib), ldab-1)
                    CALL DSYRK('Lower', 'No Transpose', i3, ib, -one, work, &
                      ldwork, one, ab(1,i+kd), ldab-1)
                    DO jj = 1, ib
                      DO ii = 1, MIN(jj, i3)
                        ab(kd+1-jj+ii, jj+i-1) = work(ii, jj)
                      END DO
                    END DO
                  END IF
                END IF
              END DO
            END IF
          END IF
          RETURN
 150      CONTINUE
          RETURN
        END SUBROUTINE

        SUBROUTINE DPBTRS(uplo, n, kd, nrhs, ab, ldab, b, ldb, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, kd, ldab, ldb, n, nrhs
          REAL(rp_) :: ab(ldab, *), b(ldb, *)
          LOGICAL :: upper
          INTEGER(ip_) :: j
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: DTBSV, XERBLA2
          INTRINSIC :: MAX
          info = 0
          upper = LSAME(uplo, 'U')
          IF (.NOT. upper .AND. .NOT. LSAME(uplo,'L')) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (kd<0) THEN
            info = -3
          ELSE IF (nrhs<0) THEN
            info = -4
          ELSE IF (ldab<kd+1) THEN
            info = -6
          ELSE IF (ldb<MAX(1,n)) THEN
            info = -8
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('PBTRS', -info)
            RETURN
          END IF
          IF (n==0 .OR. nrhs==0) RETURN
          IF (upper) THEN
            DO j = 1, nrhs
              CALL DTBSV('Upper', 'Transpose', 'Non-unit', n, kd, ab, ldab, &
                b(1,j), 1_ip_)
              CALL DTBSV('Upper', 'No transpose', 'Non-unit', n, kd, ab,    &
                ldab, b(1,j), 1_ip_)
            END DO
          ELSE
            DO j = 1, nrhs
              CALL DTBSV('Lower', 'No transpose', 'Non-unit', n, kd, ab,    &
                ldab, b(1,j), 1_ip_)
              CALL DTBSV('Lower', 'Transpose', 'Non-unit', n, kd, ab, ldab, &
                b(1,j), 1_ip_)
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DPOTF2(uplo, n, a, lda, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, lda, n
          REAL(rp_) :: a(lda, *)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          LOGICAL :: upper
          INTEGER(ip_) :: j
          REAL(rp_) :: ajj
          LOGICAL :: LISNAN
          LOGICAL :: LSAME, DISNAN
          REAL(rp_) :: DDOT
          EXTERNAL :: LSAME, DDOT, DISNAN
          EXTERNAL :: DGEMV, DSCAL, XERBLA2
          INTRINSIC :: MAX, SQRT
          info = 0
          upper = LSAME(uplo, 'U')
          IF (.NOT. upper .AND. .NOT. LSAME(uplo,'L')) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,n)) THEN
            info = -4
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('POTF2', -info)
            RETURN
          END IF
          IF (n==0) RETURN
          IF (upper) THEN
            DO j = 1, n
              ajj = a(j, j) - DDOT(j-1, a(1,j), 1_ip_, a(1,j), 1_ip_)
              LISNAN = DISNAN(ajj)
              IF (ajj<=zero .OR. LISNAN) THEN
                a(j, j) = ajj
                GO TO 30
              END IF
              ajj = SQRT(ajj)
              a(j, j) = ajj
              IF (j<n) THEN
                CALL DGEMV('Transpose', j-1, n-j, -one, a(1,j+1), lda, a(1, &
                  j), 1_ip_, one, a(j,j+1), lda)
                CALL DSCAL(n-j, one/ajj, a(j,j+1), lda)
              END IF
            END DO
          ELSE
            DO j = 1, n
              ajj = a(j, j) - DDOT(j-1, a(j,1), lda, a(j,1), lda)
              LISNAN = DISNAN(ajj)
              IF (ajj<=zero .OR. LISNAN) THEN
                a(j, j) = ajj
                GO TO 30
              END IF
              ajj = SQRT(ajj)
              a(j, j) = ajj
              IF (j<n) THEN
                CALL DGEMV('No transpose', n-j, j-1, -one, a(j+1,1), lda,   &
                  a(j,1), lda, one, a(j+1,j), 1_ip_)
                CALL DSCAL(n-j, one/ajj, a(j+1,j), 1_ip_)
              END IF
            END DO
          END IF
          GO TO 40
 30       CONTINUE
          info = j
 40       CONTINUE
          RETURN
        END SUBROUTINE

        SUBROUTINE DPOTRF(uplo, n, a, lda, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, lda, n
          REAL(rp_) :: a(lda, *)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          LOGICAL :: upper
          INTEGER(ip_) :: j, jb, nb
          LOGICAL :: LSAME
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: LSAME, ILAENV2
          EXTERNAL :: DGEMM, DPOTRF2, DSYRK, DTRSM, XERBLA2
          INTRINSIC :: MAX, MIN
          info = 0
          upper = LSAME(uplo, 'U')
          IF (.NOT. upper .AND. .NOT. LSAME(uplo,'L')) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,n)) THEN
            info = -4
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('POTRF', -info)
            RETURN
          END IF
          IF (n==0) RETURN
          nb = ILAENV2(1_ip_, 'POTRF', uplo, n, -1_ip_, -1_ip_, -1_ip_)
          IF (nb<=1 .OR. nb>=n) THEN
            CALL DPOTRF2(uplo, n, a, lda, info)
          ELSE
            IF (upper) THEN
              DO j = 1, n, nb
                jb = MIN(nb, n-j+1)
                CALL DSYRK('Upper', 'Transpose', jb, j-1, -one, a(1,j),     &
                  lda, one, a(j,j), lda)
                CALL DPOTRF2('Upper', jb, a(j,j), lda, info)
                IF (info/=0) GO TO 30
                IF (j+jb<=n) THEN
                  CALL DGEMM('Transpose', 'No transpose', jb, n-j-jb+1,     &
                    j-1, -one, a(1,j), lda, a(1,j+jb), lda, one, a(j,j+jb), &
                    lda)
                  CALL DTRSM('Left', 'Upper', 'Transpose', 'Non-unit', jb,  &
                    n-j-jb+1, one, a(j,j), lda, a(j,j+jb), lda)
                END IF
              END DO
            ELSE
              DO j = 1, n, nb
                jb = MIN(nb, n-j+1)
                CALL DSYRK('Lower', 'No transpose', jb, j-1, -one, a(j,1),  &
                  lda, one, a(j,j), lda)
                CALL DPOTRF2('Lower', jb, a(j,j), lda, info)
                IF (info/=0) GO TO 30
                IF (j+jb<=n) THEN
                  CALL DGEMM('No transpose', 'Transpose', n-j-jb+1, jb,     &
                    j-1, -one, a(j+jb,1), lda, a(j,1), lda, one, a(j+jb,j), &
                    lda)
                  CALL DTRSM('Right', 'Lower', 'Transpose', 'Non-unit',     &
                    n-j-jb+1, jb, one, a(j,j), lda, a(j+jb,j), lda)
                END IF
              END DO
            END IF
          END IF
          GO TO 40
 30       CONTINUE
          info = info + j - 1
 40       CONTINUE
          RETURN
        END SUBROUTINE

        RECURSIVE SUBROUTINE DPOTRF2(uplo, n, a, lda, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, lda, n
          REAL(rp_) :: a(lda, *)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          LOGICAL :: upper
          INTEGER(ip_) :: n1, n2, iinfo
          LOGICAL :: LISNAN
          LOGICAL :: LSAME, DISNAN
          EXTERNAL :: LSAME, DISNAN
          EXTERNAL :: DSYRK, DTRSM, XERBLA2
          INTRINSIC :: MAX, SQRT
          info = 0
          upper = LSAME(uplo, 'U')
          IF (.NOT. upper .AND. .NOT. LSAME(uplo,'L')) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,n)) THEN
            info = -4
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('POTRF2', -info)
            RETURN
          END IF
          IF (n==0) RETURN
          IF (n==1) THEN
            LISNAN = DISNAN(a(1,1))
            IF (a(1,1)<=zero .OR. LISNAN) THEN
              info = 1
              RETURN
            END IF
            a(1, 1_ip_) = SQRT(a(1,1))
          ELSE
            n1 = n/2
            n2 = n - n1
            CALL DPOTRF2(uplo, n1, a(1,1), lda, iinfo)
            IF (iinfo/=0) THEN
              info = iinfo
              RETURN
            END IF
            IF (upper) THEN
              CALL DTRSM('L', 'U', 'T', 'N', n1, n2, one, a(1,1), lda, a(1, &
                n1+1), lda)
              CALL DSYRK(uplo, 'T', n2, n1, -one, a(1,n1+1), lda, one,      &
                a(n1+1,n1+1), lda)
              CALL DPOTRF2(uplo, n2, a(n1+1,n1+1), lda, iinfo)
              IF (iinfo/=0) THEN
                info = iinfo + n1
                RETURN
              END IF
            ELSE
              CALL DTRSM('R', 'L', 'T', 'N', n2, n1, one, a(1,1), lda,      &
                a(n1+1,1), lda)
              CALL DSYRK(uplo, 'N', n2, n1, -one, a(n1+1,1), lda, one,      &
                a(n1+1,n1+1), lda)
              CALL DPOTRF2(uplo, n2, a(n1+1,n1+1), lda, iinfo)
              IF (iinfo/=0) THEN
                info = iinfo + n1
                RETURN
              END IF
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DPOTRS(uplo, n, nrhs, a, lda, b, ldb, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, lda, ldb, n, nrhs
          REAL(rp_) :: a(lda, *), b(ldb, *)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          LOGICAL :: upper
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: DTRSM, XERBLA2
          INTRINSIC :: MAX
          info = 0
          upper = LSAME(uplo, 'U')
          IF (.NOT. upper .AND. .NOT. LSAME(uplo,'L')) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (nrhs<0) THEN
            info = -3
          ELSE IF (lda<MAX(1,n)) THEN
            info = -5
          ELSE IF (ldb<MAX(1,n)) THEN
            info = -7
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('POTRS', -info)
            RETURN
          END IF
          IF (n==0 .OR. nrhs==0) RETURN
          IF (upper) THEN
            CALL DTRSM('Left', 'Upper', 'Transpose', 'Non-unit', n, nrhs,   &
              one, a, lda, b, ldb)
            CALL DTRSM('Left', 'Upper', 'No transpose', 'Non-unit', n,      &
              nrhs, one, a, lda, b, ldb)
          ELSE
            CALL DTRSM('Left', 'Lower', 'No transpose', 'Non-unit', n,      &
              nrhs, one, a, lda, b, ldb)
            CALL DTRSM('Left', 'Lower', 'Transpose', 'Non-unit', n, nrhs,   &
              one, a, lda, b, ldb)
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DPTTRF(n, d, e, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, n
          REAL(rp_) :: d(*), e(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          INTEGER(ip_) :: i, i4
          REAL(rp_) :: ei
          EXTERNAL :: XERBLA2
          INTRINSIC :: MOD
          info = 0
          IF (n<0) THEN
            info = -1
            CALL XERBLA2('PTTRF', -info)
            RETURN
          END IF
          IF (n==0) RETURN
          i4 = MOD(n-1, 4_ip_)
          DO i = 1, i4
            IF (d(i)<=zero) THEN
              info = i
              GO TO 30
            END IF
            ei = e(i)
            e(i) = ei/d(i)
            d(i+1) = d(i+1) - e(i)*ei
          END DO
          DO i = i4 + 1, n - 4, 4
            IF (d(i)<=zero) THEN
              info = i
              GO TO 30
            END IF
            ei = e(i)
            e(i) = ei/d(i)
            d(i+1) = d(i+1) - e(i)*ei
            IF (d(i+1)<=zero) THEN
              info = i + 1
              GO TO 30
            END IF
            ei = e(i+1)
            e(i+1) = ei/d(i+1)
            d(i+2) = d(i+2) - e(i+1)*ei
            IF (d(i+2)<=zero) THEN
              info = i + 2
              GO TO 30
            END IF
            ei = e(i+2)
            e(i+2) = ei/d(i+2)
            d(i+3) = d(i+3) - e(i+2)*ei
            IF (d(i+3)<=zero) THEN
              info = i + 3
              GO TO 30
            END IF
            ei = e(i+3)
            e(i+3) = ei/d(i+3)
            d(i+4) = d(i+4) - e(i+3)*ei
          END DO
          IF (d(n)<=zero) info = n
 30       CONTINUE
          RETURN
        END SUBROUTINE

        SUBROUTINE DPTTRS(n, nrhs, d, e, b, ldb, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, ldb, n, nrhs
          REAL(rp_) :: b(ldb, *), d(*), e(*)
          INTEGER(ip_) :: j, jb, nb
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: ILAENV2
          EXTERNAL :: DPTTS2, XERBLA2
          INTRINSIC :: MAX, MIN
          info = 0
          IF (n<0) THEN
            info = -1
          ELSE IF (nrhs<0) THEN
            info = -2
          ELSE IF (ldb<MAX(1,n)) THEN
            info = -6
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('PTTRS', -info)
            RETURN
          END IF
          IF (n==0 .OR. nrhs==0) RETURN
          IF (nrhs==1) THEN
            nb = 1
          ELSE
            nb = MAX(1_ip_, ILAENV2(1_ip_,'PTTRS',' ',n,nrhs,-1_ip_,        &
              -1_ip_))
          END IF
          IF (nb>=nrhs) THEN
            CALL DPTTS2(n, nrhs, d, e, b, ldb)
          ELSE
            DO j = 1, nrhs, nb
              jb = MIN(nrhs-j+1, nb)
              CALL DPTTS2(n, jb, d, e, b(1,j), ldb)
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DPTTS2(n, nrhs, d, e, b, ldb)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: ldb, n, nrhs
          REAL(rp_) :: b(ldb, *), d(*), e(*)
          INTEGER(ip_) :: i, j
          EXTERNAL :: DSCAL
          IF (n<=1) THEN
            IF (n==1) CALL DSCAL(nrhs, 1.0_rp_/d(1), b, ldb)
            RETURN
          END IF
          DO j = 1, nrhs
            DO i = 2, n
              b(i, j) = b(i, j) - b(i-1, j)*e(i-1)
            END DO
            b(n, j) = b(n, j)/d(n)
            DO i = n - 1, 1_ip_, -1_ip_
              b(i, j) = b(i, j)/d(i) - b(i+1, j)*e(i)
            END DO
          END DO
          RETURN
        END SUBROUTINE

        SUBROUTINE DRSCL(n, sa, sx, incx)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: incx, n
          REAL(rp_) :: sa
          REAL(rp_) :: sx(*)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          LOGICAL :: done
          REAL(rp_) :: bignum, cden, cden1, cnum, cnum1, mul, smlnum
          REAL(rp_) :: DLAMCH
          EXTERNAL :: DLAMCH
          EXTERNAL :: DSCAL, DLABAD
          INTRINSIC :: ABS
          IF (n<=0) RETURN
          smlnum = DLAMCH('S')
          bignum = one/smlnum
          CALL DLABAD(smlnum, bignum)
          cden = sa
          cnum = one
 10       CONTINUE
          cden1 = cden*smlnum
          cnum1 = cnum/bignum
          IF (ABS(cden1)>ABS(cnum) .AND. cnum/=zero) THEN
            mul = smlnum
            done = .FALSE.
            cden = cden1
          ELSE IF (ABS(cnum1)>ABS(cden)) THEN
            mul = bignum
            done = .FALSE.
            cnum = cnum1
          ELSE
            mul = cnum/cden
            done = .TRUE.
          END IF
          CALL DSCAL(n, mul, sx, incx)
          IF (.NOT. done) GO TO 10
          RETURN
        END SUBROUTINE

        SUBROUTINE DSTEQR(compz, n, d, e, z, ldz, work, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: compz
          INTEGER(ip_) :: info, ldz, n
          REAL(rp_) :: d(*), e(*), work(*), z(ldz, *)
          REAL(rp_) :: zero, one, two, three
          PARAMETER (zero=0.0_rp_, one=1.0_rp_, two=2.0_rp_, three=3.0_rp_)
          INTEGER(ip_) :: maxit
          PARAMETER (maxit=30)
          INTEGER(ip_) :: i, icompz, ii, iscale, j, jtot, k, l, l1, lend,   &
            lendm1, lendp1, lendsv, lm1, lsv, m, mm, mm1, nm1, nmaxit
          REAL(rp_) :: anorm, b, c, eps, eps2, f, g, p, r, rt1, rt2, s,     &
            safmax, safmin, ssfmax, ssfmin, tst
          LOGICAL :: LSAME
          REAL(rp_) :: DLAMCH, DLANST, DLAPY2
          EXTERNAL :: LSAME, DLAMCH, DLANST, DLAPY2
          EXTERNAL :: DLAE2, DLAEV2, DLARTG, DLASCL, &
            DLASET, DLASR, DLASRT, DSWAP, XERBLA2
          INTRINSIC :: ABS, MAX, SIGN, SQRT
          info = 0
          IF (LSAME(compz,'N')) THEN
            icompz = 0
          ELSE IF (LSAME(compz,'V')) THEN
            icompz = 1
          ELSE IF (LSAME(compz,'I')) THEN
            icompz = 2
          ELSE
            icompz = -1
          END IF
          IF (icompz<0) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF ((ldz<1) .OR. (icompz>0 .AND. ldz<MAX(1,n))) THEN
            info = -6
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('STEQR', -info)
            RETURN
          END IF
          IF (n==0) RETURN
          IF (n==1) THEN
            IF (icompz==2) z(1, 1_ip_) = one
            RETURN
          END IF
          eps = DLAMCH('E')
          eps2 = eps**2
          safmin = DLAMCH('S')
          safmax = one/safmin
          ssfmax = SQRT(safmax)/three
          ssfmin = SQRT(safmin)/eps2
          IF (icompz==2) CALL DLASET('Full', n, n, zero, one, z, ldz)
          nmaxit = n*maxit
          jtot = 0
          l1 = 1
          nm1 = n - 1
 10       CONTINUE
          IF (l1>n) GO TO 160
          IF (l1>1) e(l1-1) = zero
          IF (l1<=nm1) THEN
            DO m = l1, nm1
              tst = ABS(e(m))
              IF (tst==zero) GO TO 30
              IF (tst<=(SQRT(ABS(d(m)))*SQRT(ABS(d(m+1))))*eps) THEN
                e(m) = zero
                GO TO 30
              END IF
            END DO
          END IF
          m = n
 30       CONTINUE
          l = l1
          lsv = l
          lend = m
          lendsv = lend
          l1 = m + 1
          IF (lend==l) GO TO 10
          anorm = DLANST('M', lend-l+1, d(l), e(l))
          iscale = 0
          IF (anorm==zero) GO TO 10
          IF (anorm>ssfmax) THEN
            iscale = 1
            CALL DLASCL('G', 0_ip_, 0_ip_, anorm, ssfmax, lend-l+1, 1_ip_,  &
              d(l), n, info)
            CALL DLASCL('G', 0_ip_, 0_ip_, anorm, ssfmax, lend-l, 1_ip_,    &
              e(l), n, info)
          ELSE IF (anorm<ssfmin) THEN
            iscale = 2
            CALL DLASCL('G', 0_ip_, 0_ip_, anorm, ssfmin, lend-l+1, 1_ip_,  &
              d(l), n, info)
            CALL DLASCL('G', 0_ip_, 0_ip_, anorm, ssfmin, lend-l, 1_ip_,    &
              e(l), n, info)
          END IF
          IF (ABS(d(lend))<ABS(d(l))) THEN
            lend = lsv
            l = lendsv
          END IF
          IF (lend>l) THEN
 40         CONTINUE
            IF (l/=lend) THEN
              lendm1 = lend - 1
              DO m = l, lendm1
                tst = ABS(e(m))**2
                IF (tst<=(eps2*ABS(d(m)))*ABS(d(m+1))+safmin) GO TO 60
              END DO
            END IF
            m = lend
 60         CONTINUE
            IF (m<lend) e(m) = zero
            p = d(l)
            IF (m==l) GO TO 80
            IF (m==l+1) THEN
              IF (icompz>0) THEN
                CALL DLAEV2(d(l), e(l), d(l+1), rt1, rt2, c, s)
                work(l) = c
                work(n-1+l) = s
                CALL DLASR('R', 'V', 'B', n, 2_ip_, work(l), work(n-1+l),   &
                  z(1,l), ldz)
              ELSE
                CALL DLAE2(d(l), e(l), d(l+1), rt1, rt2)
              END IF
              d(l) = rt1
              d(l+1) = rt2
              e(l) = zero
              l = l + 2
              IF (l<=lend) GO TO 40
              GO TO 140
            END IF
            IF (jtot==nmaxit) GO TO 140
            jtot = jtot + 1
            g = (d(l+1)-p)/(two*e(l))
            r = DLAPY2(g, one)
            g = d(m) - p + (e(l)/(g+SIGN(r,g)))
            s = one
            c = one
            p = zero
            mm1 = m - 1
            DO i = mm1, l, -1_ip_
              f = s*e(i)
              b = c*e(i)
              CALL DLARTG(g, f, c, s, r)
              IF (i/=m-1) e(i+1) = r
              g = d(i+1) - p
              r = (d(i)-g)*s + two*c*b
              p = s*r
              d(i+1) = g + p
              g = c*r - b
              IF (icompz>0) THEN
                work(i) = c
                work(n-1+i) = -s
              END IF
            END DO
            IF (icompz>0) THEN
              mm = m - l + 1
              CALL DLASR('R', 'V', 'B', n, mm, work(l), work(n-1+l), z(1,   &
                l), ldz)
            END IF
            d(l) = d(l) - p
            e(l) = g
            GO TO 40
 80         CONTINUE
            d(l) = p
            l = l + 1
            IF (l<=lend) GO TO 40
            GO TO 140
          ELSE
 90         CONTINUE
            IF (l/=lend) THEN
              lendp1 = lend + 1
              DO m = l, lendp1, -1_ip_
                tst = ABS(e(m-1))**2
                IF (tst<=(eps2*ABS(d(m)))*ABS(d(m-1))+safmin) GO TO 110
              END DO
            END IF
            m = lend
 110        CONTINUE
            IF (m>lend) e(m-1) = zero
            p = d(l)
            IF (m==l) GO TO 130
            IF (m==l-1) THEN
              IF (icompz>0) THEN
                CALL DLAEV2(d(l-1), e(l-1), d(l), rt1, rt2, c, s)
                work(m) = c
                work(n-1+m) = s
                CALL DLASR('R', 'V', 'F', n, 2_ip_, work(m), work(n-1+m),   &
                  z(1,l-1), ldz)
              ELSE
                CALL DLAE2(d(l-1), e(l-1), d(l), rt1, rt2)
              END IF
              d(l-1) = rt1
              d(l) = rt2
              e(l-1) = zero
              l = l - 2
              IF (l>=lend) GO TO 90
              GO TO 140
            END IF
            IF (jtot==nmaxit) GO TO 140
            jtot = jtot + 1
            g = (d(l-1)-p)/(two*e(l-1))
            r = DLAPY2(g, one)
            g = d(m) - p + (e(l-1)/(g+SIGN(r,g)))
            s = one
            c = one
            p = zero
            lm1 = l - 1
            DO i = m, lm1
              f = s*e(i)
              b = c*e(i)
              CALL DLARTG(g, f, c, s, r)
              IF (i/=m) e(i-1) = r
              g = d(i) - p
              r = (d(i+1)-g)*s + two*c*b
              p = s*r
              d(i) = g + p
              g = c*r - b
              IF (icompz>0) THEN
                work(i) = c
                work(n-1+i) = s
              END IF
            END DO
            IF (icompz>0) THEN
              mm = l - m + 1
              CALL DLASR('R', 'V', 'F', n, mm, work(m), work(n-1+m), z(1,   &
                m), ldz)
            END IF
            d(l) = d(l) - p
            e(lm1) = g
            GO TO 90
 130        CONTINUE
            d(l) = p
            l = l - 1
            IF (l>=lend) GO TO 90
            GO TO 140
          END IF
 140      CONTINUE
          IF (iscale==1) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, ssfmax, anorm, lendsv-lsv+1,     &
              1_ip_, d(lsv), n, info)
            CALL DLASCL('G', 0_ip_, 0_ip_, ssfmax, anorm, lendsv-lsv,       &
              1_ip_, e(lsv), n, info)
          ELSE IF (iscale==2) THEN
            CALL DLASCL('G', 0_ip_, 0_ip_, ssfmin, anorm, lendsv-lsv+1,     &
              1_ip_, d(lsv), n, info)
            CALL DLASCL('G', 0_ip_, 0_ip_, ssfmin, anorm, lendsv-lsv,       &
              1_ip_, e(lsv), n, info)
          END IF
          IF (jtot<nmaxit) GO TO 10
          DO i = 1, n - 1
            IF (e(i)/=zero) info = info + 1
          END DO
          GO TO 190
 160      CONTINUE
          IF (icompz==0) THEN
            CALL DLASRT('I', n, d, info)
          ELSE
            DO ii = 2, n
              i = ii - 1
              k = i
              p = d(i)
              DO j = ii, n
                IF (d(j)<p) THEN
                  k = j
                  p = d(j)
                END IF
              END DO
              IF (k/=i) THEN
                d(k) = d(i)
                d(i) = p
                CALL DSWAP(n, z(1,i), 1_ip_, z(1,k), 1_ip_)
              END IF
            END DO
          END IF
 190      CONTINUE
          RETURN
        END SUBROUTINE

        SUBROUTINE DSTERF(n, d, e, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, n
          REAL(rp_) :: d(*), e(*)
          REAL(rp_) :: zero, one, two, three
          PARAMETER (zero=0.0_rp_, one=1.0_rp_, two=2.0_rp_, three=3.0_rp_)
          INTEGER(ip_) :: maxit
          PARAMETER (maxit=30)
          INTEGER(ip_) :: i, iscale, jtot, l, l1, lend, lendsv, lsv, m,     &
            nmaxit
          REAL(rp_) :: alpha, anorm, bb, c, eps, eps2, gamma, oldc, oldgam, &
            p, r, rt1, rt2, rte, s, safmax, safmin, sigma, ssfmax, ssfmin,  &
            rmax
          REAL(rp_) :: DLAMCH, DLANST, DLAPY2
          EXTERNAL :: DLAMCH, DLANST, DLAPY2
          EXTERNAL :: DLAE2, DLASCL, DLASRT, XERBLA2
          INTRINSIC :: ABS, SIGN, SQRT
          info = 0
          IF (n<0) THEN
            info = -1
            CALL XERBLA2('STERF', -info)
            RETURN
          END IF
          IF (n<=1) RETURN
          eps = DLAMCH('E')
          eps2 = eps**2
          safmin = DLAMCH('S')
          safmax = one/safmin
          ssfmax = SQRT(safmax)/three
          ssfmin = SQRT(safmin)/eps2
          rmax = DLAMCH('O')
          nmaxit = n*maxit
          sigma = zero
          jtot = 0
          l1 = 1
 10       CONTINUE
          IF (l1>n) GO TO 170
          IF (l1>1) e(l1-1) = zero
          DO m = l1, n - 1
            IF (ABS(e(m))<=(SQRT(ABS(d(m)))*SQRT(ABS(d(m+1))))*eps) THEN
              e(m) = zero
              GO TO 30
            END IF
          END DO
          m = n
 30       CONTINUE
          l = l1
          lsv = l
          lend = m
          lendsv = lend
          l1 = m + 1
          IF (lend==l) GO TO 10
          anorm = DLANST('M', lend-l+1, d(l), e(l))
          iscale = 0
          IF (anorm==zero) GO TO 10
          IF ((anorm>ssfmax)) THEN
            iscale = 1
            CALL DLASCL('G', 0_ip_, 0_ip_, anorm, ssfmax, lend-l+1, 1_ip_,  &
              d(l), n, info)
            CALL DLASCL('G', 0_ip_, 0_ip_, anorm, ssfmax, lend-l, 1_ip_,    &
              e(l), n, info)
          ELSE IF (anorm<ssfmin) THEN
            iscale = 2
            CALL DLASCL('G', 0_ip_, 0_ip_, anorm, ssfmin, lend-l+1, 1_ip_,  &
              d(l), n, info)
            CALL DLASCL('G', 0_ip_, 0_ip_, anorm, ssfmin, lend-l, 1_ip_,    &
              e(l), n, info)
          END IF
          DO i = l, lend - 1
            e(i) = e(i)**2
          END DO
          IF (ABS(d(lend))<ABS(d(l))) THEN
            lend = lsv
            l = lendsv
          END IF
          IF (lend>=l) THEN
 50         CONTINUE
            IF (l/=lend) THEN
              DO m = l, lend - 1
                IF (ABS(e(m))<=eps2*ABS(d(m)*d(m+1))) GO TO 70
              END DO
            END IF
            m = lend
 70         CONTINUE
            IF (m<lend) e(m) = zero
            p = d(l)
            IF (m==l) GO TO 90
            IF (m==l+1) THEN
              rte = SQRT(e(l))
              CALL DLAE2(d(l), rte, d(l+1), rt1, rt2)
              d(l) = rt1
              d(l+1) = rt2
              e(l) = zero
              l = l + 2
              IF (l<=lend) GO TO 50
              GO TO 150
            END IF
            IF (jtot==nmaxit) GO TO 150
            jtot = jtot + 1
            rte = SQRT(e(l))
            sigma = (d(l+1)-p)/(two*rte)
            r = DLAPY2(sigma, one)
            sigma = p - (rte/(sigma+SIGN(r,sigma)))
            c = one
            s = zero
            gamma = d(m) - sigma
            p = gamma*gamma
            DO i = m - 1, l, -1_ip_
              bb = e(i)
              r = p + bb
              IF (i/=m-1) e(i+1) = s*r
              oldc = c
              c = p/r
              s = bb/r
              oldgam = gamma
              alpha = d(i)
              gamma = c*(alpha-sigma) - s*oldgam
              d(i+1) = oldgam + (alpha-gamma)
              IF (c/=zero) THEN
                p = (gamma*gamma)/c
              ELSE
                p = oldc*bb
              END IF
            END DO
            e(l) = s*p
            d(l) = sigma + gamma
            GO TO 50
 90         CONTINUE
            d(l) = p
            l = l + 1
            IF (l<=lend) GO TO 50
            GO TO 150
          ELSE
 100        CONTINUE
            DO m = l, lend + 1, -1_ip_
              IF (ABS(e(m-1))<=eps2*ABS(d(m)*d(m-1))) GO TO 120
            END DO
            m = lend
 120        CONTINUE
            IF (m>lend) e(m-1) = zero
            p = d(l)
            IF (m==l) GO TO 140
            IF (m==l-1) THEN
              rte = SQRT(e(l-1))
              CALL DLAE2(d(l), rte, d(l-1), rt1, rt2)
              d(l) = rt1
              d(l-1) = rt2
              e(l-1) = zero
              l = l - 2
              IF (l>=lend) GO TO 100
              GO TO 150
            END IF
            IF (jtot==nmaxit) GO TO 150
            jtot = jtot + 1
            rte = SQRT(e(l-1))
            sigma = (d(l-1)-p)/(two*rte)
            r = DLAPY2(sigma, one)
            sigma = p - (rte/(sigma+SIGN(r,sigma)))
            c = one
            s = zero
            gamma = d(m) - sigma
            p = gamma*gamma
            DO i = m, l - 1
              bb = e(i)
              r = p + bb
              IF (i/=m) e(i-1) = s*r
              oldc = c
              c = p/r
              s = bb/r
              oldgam = gamma
              alpha = d(i+1)
              gamma = c*(alpha-sigma) - s*oldgam
              d(i) = oldgam + (alpha-gamma)
              IF (c/=zero) THEN
                p = (gamma*gamma)/c
              ELSE
                p = oldc*bb
              END IF
            END DO
            e(l-1) = s*p
            d(l) = sigma + gamma
            GO TO 100
 140        CONTINUE
            d(l) = p
            l = l - 1
            IF (l>=lend) GO TO 100
            GO TO 150
          END IF
 150      CONTINUE
          IF (iscale==1) CALL DLASCL('G', 0_ip_, 0_ip_, ssfmax, anorm,      &
            lendsv-lsv+1, 1_ip_, d(lsv), n, info)
          IF (iscale==2) CALL DLASCL('G', 0_ip_, 0_ip_, ssfmin, anorm,      &
            lendsv-lsv+1, 1_ip_, d(lsv), n, info)
          IF (jtot<nmaxit) GO TO 10
          DO i = 1, n - 1
            IF (e(i)/=zero) info = info + 1
          END DO
          GO TO 180
 170      CONTINUE
          CALL DLASRT('I', n, d, info)
 180      CONTINUE
          RETURN
        END SUBROUTINE

        SUBROUTINE DSYEV(jobz, uplo, n, a, lda, w, work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: jobz, uplo
          INTEGER(ip_) :: info, lda, lwork, n
          REAL(rp_) :: a(lda, *), w(*), work(*)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          LOGICAL :: lower, lquery, wantz
          INTEGER(ip_) :: iinfo, imax, inde, indtau, indwrk, iscale,        &
            llwork, lwkopt, nb
          REAL(rp_) :: anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum
          LOGICAL :: LSAME
          INTEGER(ip_) :: ILAENV2
          REAL(rp_) :: DLAMCH, DLANSY
          EXTERNAL :: LSAME, ILAENV2, DLAMCH, DLANSY
          EXTERNAL :: DLASCL, DORGTR, DSCAL, DSTEQR, &
            DSTERF, DSYTRD, XERBLA2
          INTRINSIC :: MAX, SQRT
          wantz = LSAME(jobz, 'V')
          lower = LSAME(uplo, 'L')
          lquery = (lwork==-1)
          info = 0
          IF (.NOT. (wantz .OR. LSAME(jobz,'N'))) THEN
            info = -1
          ELSE IF (.NOT. (lower .OR. LSAME(uplo,'U'))) THEN
            info = -2
          ELSE IF (n<0) THEN
            info = -3
          ELSE IF (lda<MAX(1,n)) THEN
            info = -5
          END IF
          IF (info==0) THEN
            nb = ILAENV2(1_ip_, 'SYTRD', uplo, n, -1_ip_, -1_ip_, -1_ip_)
            lwkopt = MAX(1, (nb+2)*n)
            work(1) = REAL(lwkopt,rp_)
            IF (lwork<MAX(1,3*n-1) .AND. .NOT. lquery) info = -8
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('SYEV ', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          IF (n==0) THEN
            RETURN
          END IF
          IF (n==1) THEN
            w(1) = a(1, 1_ip_)
            work(1) = 2
            IF (wantz) a(1, 1_ip_) = one
            RETURN
          END IF
          safmin = DLAMCH('Safe minimum')
          eps = DLAMCH('Precision')
          smlnum = safmin/eps
          bignum = one/smlnum
          rmin = SQRT(smlnum)
          rmax = SQRT(bignum)
          anrm = DLANSY('M', uplo, n, a, lda, work)
          iscale = 0
          IF (anrm>zero .AND. anrm<rmin) THEN
            iscale = 1
            sigma = rmin/anrm
          ELSE IF (anrm>rmax) THEN
            iscale = 1
            sigma = rmax/anrm
          END IF
          IF (iscale==1) CALL DLASCL(uplo, 0_ip_, 0_ip_, one, sigma, n, n,  &
            a, lda, info)
          inde = 1
          indtau = inde + n
          indwrk = indtau + n
          llwork = lwork - indwrk + 1
          CALL DSYTRD(uplo, n, a, lda, w, work(inde), work(indtau),         &
            work(indwrk), llwork, iinfo)
          IF (.NOT. wantz) THEN
            CALL DSTERF(n, w, work(inde), info)
          ELSE
            CALL DORGTR(uplo, n, a, lda, work(indtau), work(indwrk),        &
              llwork, iinfo)
            CALL DSTEQR(jobz, n, w, work(inde), a, lda, work(indtau), info)
          END IF
          IF (iscale==1) THEN
            IF (info==0) THEN
              imax = n
            ELSE
              imax = info - 1
            END IF
            CALL DSCAL(imax, one/sigma, w, 1_ip_)
          END IF
          work(1) = REAL(lwkopt,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DSYGS2(itype, uplo, n, a, lda, b, ldb, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, itype, lda, ldb, n
          REAL(rp_) :: a(lda, *), b(ldb, *)
          REAL(rp_) :: one, half
          PARAMETER (one=1.0_rp_, half=0.5_rp_)
          LOGICAL :: upper
          INTEGER(ip_) :: k
          REAL(rp_) :: akk, bkk, ct
          EXTERNAL :: DAXPY, DSCAL, DSYR2, DTRMV, DTRSV, &
            XERBLA2
          INTRINSIC :: MAX
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          info = 0
          upper = LSAME(uplo, 'U')
          IF (itype<1 .OR. itype>3) THEN
            info = -1
          ELSE IF (.NOT. upper .AND. .NOT. LSAME(uplo,'L')) THEN
            info = -2
          ELSE IF (n<0) THEN
            info = -3
          ELSE IF (lda<MAX(1,n)) THEN
            info = -5
          ELSE IF (ldb<MAX(1,n)) THEN
            info = -7
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('SYGS2', -info)
            RETURN
          END IF
          IF (itype==1) THEN
            IF (upper) THEN
              DO k = 1, n
                akk = a(k, k)
                bkk = b(k, k)
                akk = akk/bkk**2
                a(k, k) = akk
                IF (k<n) THEN
                  CALL DSCAL(n-k, one/bkk, a(k,k+1), lda)
                  ct = -half*akk
                  CALL DAXPY(n-k, ct, b(k,k+1), ldb, a(k,k+1), lda)
                  CALL DSYR2(uplo, n-k, -one, a(k,k+1), lda, b(k,k+1), ldb, &
                    a(k+1,k+1), lda)
                  CALL DAXPY(n-k, ct, b(k,k+1), ldb, a(k,k+1), lda)
                  CALL DTRSV(uplo, 'Transpose', 'Non-unit', n-k, b(k+1,     &
                    k+1), ldb, a(k,k+1), lda)
                END IF
              END DO
            ELSE
              DO k = 1, n
                akk = a(k, k)
                bkk = b(k, k)
                akk = akk/bkk**2
                a(k, k) = akk
                IF (k<n) THEN
                  CALL DSCAL(n-k, one/bkk, a(k+1,k), 1_ip_)
                  ct = -half*akk
                  CALL DAXPY(n-k, ct, b(k+1,k), 1_ip_, a(k+1,k), 1_ip_)
                  CALL DSYR2(uplo, n-k, -one, a(k+1,k), 1_ip_, b(k+1,k),    &
                    1_ip_, a(k+1,k+1), lda)
                  CALL DAXPY(n-k, ct, b(k+1,k), 1_ip_, a(k+1,k), 1_ip_)
                  CALL DTRSV(uplo, 'No transpose', 'Non-unit', n-k, b(k+1,  &
                    k+1), ldb, a(k+1,k), 1_ip_)
                END IF
              END DO
            END IF
          ELSE
            IF (upper) THEN
              DO k = 1, n
                akk = a(k, k)
                bkk = b(k, k)
                CALL DTRMV(uplo, 'No transpose', 'Non-unit', k-1, b, ldb,   &
                  a(1,k), 1_ip_)
                ct = half*akk
                CALL DAXPY(k-1, ct, b(1,k), 1_ip_, a(1,k), 1_ip_)
                CALL DSYR2(uplo, k-1, one, a(1,k), 1_ip_, b(1,k), 1_ip_, a, &
                  lda)
                CALL DAXPY(k-1, ct, b(1,k), 1_ip_, a(1,k), 1_ip_)
                CALL DSCAL(k-1, bkk, a(1,k), 1_ip_)
                a(k, k) = akk*bkk**2
              END DO
            ELSE
              DO k = 1, n
                akk = a(k, k)
                bkk = b(k, k)
                CALL DTRMV(uplo, 'Transpose', 'Non-unit', k-1, b, ldb, a(k, &
                  1), lda)
                ct = half*akk
                CALL DAXPY(k-1, ct, b(k,1), ldb, a(k,1), lda)
                CALL DSYR2(uplo, k-1, one, a(k,1), lda, b(k,1), ldb, a,     &
                  lda)
                CALL DAXPY(k-1, ct, b(k,1), ldb, a(k,1), lda)
                CALL DSCAL(k-1, bkk, a(k,1), lda)
                a(k, k) = akk*bkk**2
              END DO
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DSYGST(itype, uplo, n, a, lda, b, ldb, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, itype, lda, ldb, n
          REAL(rp_) :: a(lda, *), b(ldb, *)
          REAL(rp_) :: one, half
          PARAMETER (one=1.0_rp_, half=0.5_rp_)
          LOGICAL :: upper
          INTEGER(ip_) :: k, kb, nb
          EXTERNAL :: DSYGS2, DSYMM, DSYR2K, DTRMM, DTRSM, &
            XERBLA2
          INTRINSIC :: MAX, MIN
          LOGICAL :: LSAME
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: LSAME, ILAENV2
          info = 0
          upper = LSAME(uplo, 'U')
          IF (itype<1 .OR. itype>3) THEN
            info = -1
          ELSE IF (.NOT. upper .AND. .NOT. LSAME(uplo,'L')) THEN
            info = -2
          ELSE IF (n<0) THEN
            info = -3
          ELSE IF (lda<MAX(1,n)) THEN
            info = -5
          ELSE IF (ldb<MAX(1,n)) THEN
            info = -7
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('SYGST', -info)
            RETURN
          END IF
          IF (n==0) RETURN
          nb = ILAENV2(1_ip_, 'SYGST', uplo, n, -1_ip_, -1_ip_, -1_ip_)
          IF (nb<=1 .OR. nb>=n) THEN
            CALL DSYGS2(itype, uplo, n, a, lda, b, ldb, info)
          ELSE
            IF (itype==1) THEN
              IF (upper) THEN
                DO k = 1, n, nb
                  kb = MIN(n-k+1, nb)
                  CALL DSYGS2(itype, uplo, kb, a(k,k), lda, b(k,k), ldb,    &
                    info)
                  IF (k+kb<=n) THEN
                    CALL DTRSM('Left', uplo, 'Transpose', 'Non-unit', kb,   &
                      n-k-kb+1, one, b(k,k), ldb, a(k,k+kb), lda)
                    CALL DSYMM('Left', uplo, kb, n-k-kb+1, -half, a(k,k),   &
                      lda, b(k,k+kb), ldb, one, a(k,k+kb), lda)
                    CALL DSYR2K(uplo, 'Transpose', n-k-kb+1, kb, -one, a(k, &
                      k+kb), lda, b(k,k+kb), ldb, one, a(k+kb,k+kb), lda)
                    CALL DSYMM('Left', uplo, kb, n-k-kb+1, -half, a(k,k),   &
                      lda, b(k,k+kb), ldb, one, a(k,k+kb), lda)
                    CALL DTRSM('Right', uplo, 'No transpose', 'Non-unit',   &
                      kb, n-k-kb+1, one, b(k+kb,k+kb), ldb, a(k,k+kb), lda)
                  END IF
                END DO
              ELSE
                DO k = 1, n, nb
                  kb = MIN(n-k+1, nb)
                  CALL DSYGS2(itype, uplo, kb, a(k,k), lda, b(k,k), ldb,    &
                    info)
                  IF (k+kb<=n) THEN
                    CALL DTRSM('Right', uplo, 'Transpose', 'Non-unit',      &
                      n-k-kb+1, kb, one, b(k,k), ldb, a(k+kb,k), lda)
                    CALL DSYMM('Right', uplo, n-k-kb+1, kb, -half, a(k,k),  &
                      lda, b(k+kb,k), ldb, one, a(k+kb,k), lda)
                    CALL DSYR2K(uplo, 'No transpose', n-k-kb+1, kb, -one,   &
                      a(k+kb,k), lda, b(k+kb,k), ldb, one, a(k+kb,k+kb), lda)
                    CALL DSYMM('Right', uplo, n-k-kb+1, kb, -half, a(k,k),  &
                      lda, b(k+kb,k), ldb, one, a(k+kb,k), lda)
                    CALL DTRSM('Left', uplo, 'No transpose', 'Non-unit',    &
                      n-k-kb+1, kb, one, b(k+kb,k+kb), ldb, a(k+kb,k), lda)
                  END IF
                END DO
              END IF
            ELSE
              IF (upper) THEN
                DO k = 1, n, nb
                  kb = MIN(n-k+1, nb)
                  CALL DTRMM('Left', uplo, 'No transpose', 'Non-unit', k-1, &
                    kb, one, b, ldb, a(1,k), lda)
                  CALL DSYMM('Right', uplo, k-1, kb, half, a(k,k), lda,     &
                    b(1,k), ldb, one, a(1,k), lda)
                  CALL DSYR2K(uplo, 'No transpose', k-1, kb, one, a(1,k),   &
                    lda, b(1,k), ldb, one, a, lda)
                  CALL DSYMM('Right', uplo, k-1, kb, half, a(k,k), lda,     &
                    b(1,k), ldb, one, a(1,k), lda)
                  CALL DTRMM('Right', uplo, 'Transpose', 'Non-unit', k-1,   &
                    kb, one, b(k,k), ldb, a(1,k), lda)
                  CALL DSYGS2(itype, uplo, kb, a(k,k), lda, b(k,k), ldb,    &
                    info)
                END DO
              ELSE
                DO k = 1, n, nb
                  kb = MIN(n-k+1, nb)
                  CALL DTRMM('Right', uplo, 'No transpose', 'Non-unit', kb, &
                    k-1, one, b, ldb, a(k,1), lda)
                  CALL DSYMM('Left', uplo, kb, k-1, half, a(k,k), lda, b(k, &
                    1), ldb, one, a(k,1), lda)
                  CALL DSYR2K(uplo, 'Transpose', k-1, kb, one, a(k,1), lda, &
                    b(k,1), ldb, one, a, lda)
                  CALL DSYMM('Left', uplo, kb, k-1, half, a(k,k), lda, b(k, &
                    1), ldb, one, a(k,1), lda)
                  CALL DTRMM('Left', uplo, 'Transpose', 'Non-unit', kb,     &
                    k-1, one, b(k,k), ldb, a(k,1), lda)
                  CALL DSYGS2(itype, uplo, kb, a(k,k), lda, b(k,k), ldb,    &
                    info)
                END DO
              END IF
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DSYGV(itype, jobz, uplo, n, a, lda, b, ldb, w, work,     &
          lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: jobz, uplo
          INTEGER(ip_) :: info, itype, lda, ldb, lwork, n
          REAL(rp_) :: a(lda, *), b(ldb, *), w(*), work(*)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          LOGICAL :: lquery, upper, wantz
          CHARACTER :: trans
          INTEGER(ip_) :: lwkmin, lwkopt, nb, neig
          LOGICAL :: LSAME
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: LSAME, ILAENV2
          EXTERNAL :: DPOTRF, DSYEV, DSYGST, DTRMM, DTRSM, &
            XERBLA2
          INTRINSIC :: MAX
          wantz = LSAME(jobz, 'V')
          upper = LSAME(uplo, 'U')
          lquery = (lwork==-1)
          info = 0
          IF (itype<1 .OR. itype>3) THEN
            info = -1
          ELSE IF (.NOT. (wantz .OR. LSAME(jobz,'N'))) THEN
            info = -2
          ELSE IF (.NOT. (upper .OR. LSAME(uplo,'L'))) THEN
            info = -3
          ELSE IF (n<0) THEN
            info = -4
          ELSE IF (lda<MAX(1,n)) THEN
            info = -6
          ELSE IF (ldb<MAX(1,n)) THEN
            info = -8
          END IF
          IF (info==0) THEN
            lwkmin = MAX(1, 3*n-1)
            nb = ILAENV2(1_ip_, 'SYTRD', uplo, n, -1_ip_, -1_ip_, -1_ip_)
            lwkopt = MAX(lwkmin, (nb+2)*n)
            work(1) = REAL(lwkopt,rp_)
            IF (lwork<lwkmin .AND. .NOT. lquery) THEN
              info = -11
            END IF
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('SYGV ', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          IF (n==0) RETURN
          CALL DPOTRF(uplo, n, b, ldb, info)
          IF (info/=0) THEN
            info = n + info
            RETURN
          END IF
          CALL DSYGST(itype, uplo, n, a, lda, b, ldb, info)
          CALL DSYEV(jobz, uplo, n, a, lda, w, work, lwork, info)
          IF (wantz) THEN
            neig = n
            IF (info>0) neig = info - 1
            IF (itype==1 .OR. itype==2) THEN
              IF (upper) THEN
                trans = 'N'
              ELSE
                trans = 'T'
              END IF
              CALL DTRSM('Left', uplo, trans, 'Non-unit', n, neig, one, b,  &
                ldb, a, lda)
            ELSE IF (itype==3) THEN
              IF (upper) THEN
                trans = 'T'
              ELSE
                trans = 'N'
              END IF
              CALL DTRMM('Left', uplo, trans, 'Non-unit', n, neig, one, b,  &
                ldb, a, lda)
            END IF
          END IF
          work(1) = REAL(lwkopt,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DSYTD2(uplo, n, a, lda, d, e, tau, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, lda, n
          REAL(rp_) :: a(lda, *), d(*), e(*), tau(*)
          REAL(rp_) :: one, zero, half
          PARAMETER (one=1.0_rp_, zero=0.0_rp_, half=1.0_rp_/2.0_rp_)
          LOGICAL :: upper
          INTEGER(ip_) :: i
          REAL(rp_) :: alpha, taui
          EXTERNAL :: DAXPY, DLARFG, DSYMV, DSYR2, XERBLA2
          LOGICAL :: LSAME
          REAL(rp_) :: DDOT
          EXTERNAL :: LSAME, DDOT
          INTRINSIC :: MAX, MIN
          info = 0
          upper = LSAME(uplo, 'U')
          IF (.NOT. upper .AND. .NOT. LSAME(uplo,'L')) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,n)) THEN
            info = -4
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('SYTD2', -info)
            RETURN
          END IF
          IF (n<=0) RETURN
          IF (upper) THEN
            DO i = n - 1, 1_ip_, -1_ip_
              CALL DLARFG(i, a(i,i+1), a(1,i+1), 1_ip_, taui)
              e(i) = a(i, i+1)
              IF (taui/=zero) THEN
                a(i, i+1) = one
                CALL DSYMV(uplo, i, taui, a, lda, a(1,i+1), 1_ip_, zero,    &
                  tau, 1_ip_)
                alpha = -half*taui*DDOT(i, tau, 1_ip_, a(1,i+1), 1_ip_)
                CALL DAXPY(i, alpha, a(1,i+1), 1_ip_, tau, 1_ip_)
                CALL DSYR2(uplo, i, -one, a(1,i+1), 1_ip_, tau, 1_ip_, a,   &
                  lda)
                a(i, i+1) = e(i)
              END IF
              d(i+1) = a(i+1, i+1)
              tau(i) = taui
            END DO
            d(1) = a(1, 1_ip_)
          ELSE
            DO i = 1, n - 1
              CALL DLARFG(n-i, a(i+1,i), a(MIN(i+2,n),i), 1_ip_, taui)
              e(i) = a(i+1, i)
              IF (taui/=zero) THEN
                a(i+1, i) = one
                CALL DSYMV(uplo, n-i, taui, a(i+1,i+1), lda, a(i+1,i),      &
                  1_ip_, zero, tau(i), 1_ip_)
                alpha = -half*taui*DDOT(n-i, tau(i), 1_ip_, a(i+1,i),       &
                  1_ip_)
                CALL DAXPY(n-i, alpha, a(i+1,i), 1_ip_, tau(i), 1_ip_)
                CALL DSYR2(uplo, n-i, -one, a(i+1,i), 1_ip_, tau(i), 1_ip_, &
                  a(i+1,i+1), lda)
                a(i+1, i) = e(i)
              END IF
              d(i) = a(i, i)
              tau(i) = taui
            END DO
            d(n) = a(n, n)
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DSYTF2(uplo, n, a, lda, ipiv, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, lda, n
          INTEGER(ip_) :: ipiv(*)
          REAL(rp_) :: a(lda, *)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          REAL(rp_) :: eight, sevten
          PARAMETER (eight=8.0_rp_, sevten=17.0_rp_)
          LOGICAL :: upper
          INTEGER(ip_) :: i, imax, j, jmax, k, kk, kp, kstep
          REAL(rp_) :: absakk, alpha, colmax, d11, d12, d21, d22, r1,       &
            rowmax, t, wk, wkm1, wkp1
          LOGICAL :: LSAME, DISNAN
          INTEGER(ip_) :: IDAMAX
          LOGICAL :: LISNAN
          EXTERNAL :: LSAME, IDAMAX, DISNAN
          EXTERNAL :: DSCAL, DSWAP, DSYR, XERBLA2
          INTRINSIC :: ABS, MAX, SQRT
          info = 0
          upper = LSAME(uplo, 'U')
          IF (.NOT. upper .AND. .NOT. LSAME(uplo,'L')) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,n)) THEN
            info = -4
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('SYTF2', -info)
            RETURN
          END IF
          alpha = (one+SQRT(sevten))/eight
          IF (upper) THEN
            k = n
 10         CONTINUE
            IF (k<1) GO TO 70
            kstep = 1
            absakk = ABS(a(k,k))
            IF (k>1) THEN
              imax = IDAMAX(k-1, a(1,k), 1_ip_)
              colmax = ABS(a(imax,k))
            ELSE
              colmax = zero
            END IF
            LISNAN = DISNAN(absakk)
            IF ((MAX(absakk,colmax)==zero) .OR. LISNAN) THEN
              IF (info==0) info = k
              kp = k
            ELSE
              IF (absakk>=alpha*colmax) THEN
                kp = k
              ELSE
                jmax = imax + IDAMAX(k-imax, a(imax,imax+1), lda)
                rowmax = ABS(a(imax,jmax))
                IF (imax>1) THEN
                  jmax = IDAMAX(imax-1, a(1,imax), 1_ip_)
                  rowmax = MAX(rowmax, ABS(a(jmax,imax)))
                END IF
                IF (absakk>=alpha*colmax*(colmax/rowmax)) THEN
                  kp = k
                ELSE IF (ABS(a(imax,imax))>=alpha*rowmax) THEN
                  kp = imax
                ELSE
                  kp = imax
                  kstep = 2
                END IF
              END IF
              kk = k - kstep + 1
              IF (kp/=kk) THEN
                CALL DSWAP(kp-1, a(1,kk), 1_ip_, a(1,kp), 1_ip_)
                CALL DSWAP(kk-kp-1, a(kp+1,kk), 1_ip_, a(kp,kp+1), lda)
                t = a(kk, kk)
                a(kk, kk) = a(kp, kp)
                a(kp, kp) = t
                IF (kstep==2) THEN
                  t = a(k-1, k)
                  a(k-1, k) = a(kp, k)
                  a(kp, k) = t
                END IF
              END IF
              IF (kstep==1) THEN
                r1 = one/a(k, k)
                CALL DSYR(uplo, k-1, -r1, a(1,k), 1_ip_, a, lda)
                CALL DSCAL(k-1, r1, a(1,k), 1_ip_)
              ELSE
                IF (k>2) THEN
                  d12 = a(k-1, k)
                  d22 = a(k-1, k-1)/d12
                  d11 = a(k, k)/d12
                  t = one/(d11*d22-one)
                  d12 = t/d12
                  DO j = k - 2, 1_ip_, -1_ip_
                    wkm1 = d12*(d11*a(j,k-1)-a(j,k))
                    wk = d12*(d22*a(j,k)-a(j,k-1))
                    DO i = j, 1_ip_, -1_ip_
                      a(i, j) = a(i, j) - a(i, k)*wk - a(i, k-1)*wkm1
                    END DO
                    a(j, k) = wk
                    a(j, k-1) = wkm1
                  END DO
                END IF
              END IF
            END IF
            IF (kstep==1) THEN
              ipiv(k) = kp
            ELSE
              ipiv(k) = -kp
              ipiv(k-1) = -kp
            END IF
            k = k - kstep
            GO TO 10
          ELSE
            k = 1
 40         CONTINUE
            IF (k>n) GO TO 70
            kstep = 1
            absakk = ABS(a(k,k))
            IF (k<n) THEN
              imax = k + IDAMAX(n-k, a(k+1,k), 1_ip_)
              colmax = ABS(a(imax,k))
            ELSE
              colmax = zero
            END IF
            LISNAN = DISNAN(absakk)
            IF ((MAX(absakk,colmax)==zero) .OR. DISNAN(absakk)) THEN
              IF (info==0) info = k
              kp = k
            ELSE
              IF (absakk>=alpha*colmax) THEN
                kp = k
              ELSE
                jmax = k - 1 + IDAMAX(imax-k, a(imax,k), lda)
                rowmax = ABS(a(imax,jmax))
                IF (imax<n) THEN
                  jmax = imax + IDAMAX(n-imax, a(imax+1,imax), 1_ip_)
                  rowmax = MAX(rowmax, ABS(a(jmax,imax)))
                END IF
                IF (absakk>=alpha*colmax*(colmax/rowmax)) THEN
                  kp = k
                ELSE IF (ABS(a(imax,imax))>=alpha*rowmax) THEN
                  kp = imax
                ELSE
                  kp = imax
                  kstep = 2
                END IF
              END IF
              kk = k + kstep - 1
              IF (kp/=kk) THEN
                IF (kp<n) CALL DSWAP(n-kp, a(kp+1,kk), 1_ip_, a(kp+1,kp),   &
                  1_ip_)
                CALL DSWAP(kp-kk-1, a(kk+1,kk), 1_ip_, a(kp,kk+1), lda)
                t = a(kk, kk)
                a(kk, kk) = a(kp, kp)
                a(kp, kp) = t
                IF (kstep==2) THEN
                  t = a(k+1, k)
                  a(k+1, k) = a(kp, k)
                  a(kp, k) = t
                END IF
              END IF
              IF (kstep==1) THEN
                IF (k<n) THEN
                  d11 = one/a(k, k)
                  CALL DSYR(uplo, n-k, -d11, a(k+1,k), 1_ip_, a(k+1,k+1),   &
                    lda)
                  CALL DSCAL(n-k, d11, a(k+1,k), 1_ip_)
                END IF
              ELSE
                IF (k<n-1) THEN
                  d21 = a(k+1, k)
                  d11 = a(k+1, k+1)/d21
                  d22 = a(k, k)/d21
                  t = one/(d11*d22-one)
                  d21 = t/d21
                  DO j = k + 2, n
                    wk = d21*(d11*a(j,k)-a(j,k+1))
                    wkp1 = d21*(d22*a(j,k+1)-a(j,k))
                    DO i = j, n
                      a(i, j) = a(i, j) - a(i, k)*wk - a(i, k+1)*wkp1
                    END DO
                    a(j, k) = wk
                    a(j, k+1) = wkp1
                  END DO
                END IF
              END IF
            END IF
            IF (kstep==1) THEN
              ipiv(k) = kp
            ELSE
              ipiv(k) = -kp
              ipiv(k+1) = -kp
            END IF
            k = k + kstep
            GO TO 40
          END IF
 70       CONTINUE
          RETURN
        END SUBROUTINE

        SUBROUTINE DSYTRD(uplo, n, a, lda, d, e, tau, work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, lda, lwork, n
          REAL(rp_) :: a(lda, *), d(*), e(*), tau(*), work(*)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          LOGICAL :: lquery, upper
          INTEGER(ip_) :: i, iinfo, iws, j, kk, ldwork, lwkopt, nb, nbmin,  &
            nx
          EXTERNAL :: DLATRD, DSYR2K, DSYTD2, XERBLA2
          INTRINSIC :: MAX
          LOGICAL :: LSAME
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: LSAME, ILAENV2
          info = 0
          upper = LSAME(uplo, 'U')
          lquery = (lwork==-1)
          IF (.NOT. upper .AND. .NOT. LSAME(uplo,'L')) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,n)) THEN
            info = -4
          ELSE IF (lwork<1 .AND. .NOT. lquery) THEN
            info = -9
          END IF
          IF (info==0) THEN
            nb = ILAENV2(1_ip_, 'SYTRD', uplo, n, -1_ip_, -1_ip_, -1_ip_)
            lwkopt = n*nb
            work(1) = REAL(lwkopt,rp_)
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('SYTRD', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          IF (n==0) THEN
            work(1) = 1
            RETURN
          END IF
          nx = n
          iws = 1
          IF (nb>1 .AND. nb<n) THEN
            nx = MAX(nb, ILAENV2(3_ip_,'SYTRD',uplo,n,-1_ip_,-1_ip_,        &
              -1_ip_))
            IF (nx<n) THEN
              ldwork = n
              iws = ldwork*nb
              IF (lwork<iws) THEN
                nb = MAX(lwork/ldwork, 1_ip_)
                nbmin = ILAENV2(2_ip_, 'SYTRD', uplo, n, -1_ip_, -1_ip_,    &
                  -1_ip_)
                IF (nb<nbmin) nx = n
              END IF
            ELSE
              nx = n
            END IF
          ELSE
            nb = 1
          END IF
          IF (upper) THEN
            kk = n - ((n-nx+nb-1)/nb)*nb
            DO i = n - nb + 1, kk + 1, -nb
              CALL DLATRD(uplo, i+nb-1, nb, a, lda, e, tau, work, ldwork)
              CALL DSYR2K(uplo, 'No transpose', i-1, nb, -one, a(1,i), lda, &
                work, ldwork, one, a, lda)
              DO j = i, i + nb - 1
                a(j-1, j) = e(j-1)
                d(j) = a(j, j)
              END DO
            END DO
            CALL DSYTD2(uplo, kk, a, lda, d, e, tau, iinfo)
          ELSE
            DO i = 1, n - nx, nb
              CALL DLATRD(uplo, n-i+1, nb, a(i,i), lda, e(i), tau(i), work, &
                ldwork)
              CALL DSYR2K(uplo, 'No transpose', n-i-nb+1, nb, -one, a(i+nb, &
                i), lda, work(nb+1), ldwork, one, a(i+nb,i+nb), lda)
              DO j = i, i + nb - 1
                a(j+1, j) = e(j)
                d(j) = a(j, j)
              END DO
            END DO
            CALL DSYTD2(uplo, n-i+1, a(i,i), lda, d(i), e(i), tau(i),       &
              iinfo)
          END IF
          work(1) = REAL(lwkopt,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DSYTRF(uplo, n, a, lda, ipiv, work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, lda, lwork, n
          INTEGER(ip_) :: ipiv(*)
          REAL(rp_) :: a(lda, *), work(*)
          LOGICAL :: lquery, upper
          INTEGER(ip_) :: iinfo, iws, j, k, kb, ldwork, lwkopt, nb, nbmin
          LOGICAL :: LSAME
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: LSAME, ILAENV2
          EXTERNAL :: DLASYF, DSYTF2, XERBLA2
          INTRINSIC :: MAX
          info = 0
          upper = LSAME(uplo, 'U')
          lquery = (lwork==-1)
          IF (.NOT. upper .AND. .NOT. LSAME(uplo,'L')) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (lda<MAX(1,n)) THEN
            info = -4
          ELSE IF (lwork<1 .AND. .NOT. lquery) THEN
            info = -7
          END IF
          IF (info==0) THEN
            nb = ILAENV2(1_ip_, 'SYTRF', uplo, n, -1_ip_, -1_ip_, -1_ip_)
            lwkopt = n*nb
            work(1) = REAL(lwkopt,rp_)
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('SYTRF', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          nbmin = 2
          ldwork = n
          IF (nb>1 .AND. nb<n) THEN
            iws = ldwork*nb
            IF (lwork<iws) THEN
              nb = MAX(lwork/ldwork, 1_ip_)
              nbmin = MAX(2_ip_, ILAENV2(2_ip_,'SYTRF',uplo,n,-1_ip_,       &
                -1_ip_,-1_ip_))
            END IF
          ELSE
            iws = 1
          END IF
          IF (nb<nbmin) nb = n
          IF (upper) THEN
            k = n
 10         CONTINUE
            IF (k<1) GO TO 40
            IF (k>nb) THEN
              CALL DLASYF(uplo, k, nb, kb, a, lda, ipiv, work, ldwork,      &
                iinfo)
            ELSE
              CALL DSYTF2(uplo, k, a, lda, ipiv, iinfo)
              kb = k
            END IF
            IF (info==0 .AND. iinfo>0) info = iinfo
            k = k - kb
            GO TO 10
          ELSE
            k = 1
 20         CONTINUE
            IF (k>n) GO TO 40
            IF (k<=n-nb) THEN
              CALL DLASYF(uplo, n-k+1, nb, kb, a(k,k), lda, ipiv(k), work,  &
                ldwork, iinfo)
            ELSE
              CALL DSYTF2(uplo, n-k+1, a(k,k), lda, ipiv(k), iinfo)
              kb = n - k + 1
            END IF
            IF (info==0 .AND. iinfo>0) info = iinfo + k - 1
            DO j = k, k + kb - 1
              IF (ipiv(j)>0) THEN
                ipiv(j) = ipiv(j) + k - 1
              ELSE
                ipiv(j) = ipiv(j) - k + 1
              END IF
            END DO
            k = k + kb
            GO TO 20
          END IF
 40       CONTINUE
          work(1) = REAL(lwkopt,rp_)
          RETURN
        END SUBROUTINE

        SUBROUTINE DSYTRS(uplo, n, nrhs, a, lda, ipiv, b, ldb, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: uplo
          INTEGER(ip_) :: info, lda, ldb, n, nrhs
          INTEGER(ip_) :: ipiv(*)
          REAL(rp_) :: a(lda, *), b(ldb, *)
          REAL(rp_) :: one
          PARAMETER (one=1.0_rp_)
          LOGICAL :: upper
          INTEGER(ip_) :: j, k, kp
          REAL(rp_) :: ak, akm1, akm1k, bk, bkm1, denom
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: DGEMV, DGER, DSCAL, DSWAP, XERBLA2
          INTRINSIC :: MAX
          info = 0
          upper = LSAME(uplo, 'U')
          IF (.NOT. upper .AND. .NOT. LSAME(uplo,'L')) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (nrhs<0) THEN
            info = -3
          ELSE IF (lda<MAX(1,n)) THEN
            info = -5
          ELSE IF (ldb<MAX(1,n)) THEN
            info = -8
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('SYTRS', -info)
            RETURN
          END IF
          IF (n==0 .OR. nrhs==0) RETURN
          IF (upper) THEN
            k = n
 10         CONTINUE
            IF (k<1) GO TO 30
            IF (ipiv(k)>0) THEN
              kp = ipiv(k)
              IF (kp/=k) CALL DSWAP(nrhs, b(k,1), ldb, b(kp,1), ldb)
              CALL DGER(k-1, nrhs, -one, a(1,k), 1_ip_, b(k,1), ldb, b(1,   &
                1), ldb)
              CALL DSCAL(nrhs, one/a(k,k), b(k,1), ldb)
              k = k - 1
            ELSE
              kp = -ipiv(k)
              IF (kp/=k-1) CALL DSWAP(nrhs, b(k-1,1), ldb, b(kp,1), ldb)
              CALL DGER(k-2, nrhs, -one, a(1,k), 1_ip_, b(k,1), ldb, b(1,   &
                1), ldb)
              CALL DGER(k-2, nrhs, -one, a(1,k-1), 1_ip_, b(k-1,1), ldb,    &
                b(1,1), ldb)
              akm1k = a(k-1, k)
              akm1 = a(k-1, k-1)/akm1k
              ak = a(k, k)/akm1k
              denom = akm1*ak - one
              DO j = 1, nrhs
                bkm1 = b(k-1, j)/akm1k
                bk = b(k, j)/akm1k
                b(k-1, j) = (ak*bkm1-bk)/denom
                b(k, j) = (akm1*bk-bkm1)/denom
              END DO
              k = k - 2
            END IF
            GO TO 10
 30         CONTINUE
            k = 1
 40         CONTINUE
            IF (k>n) GO TO 50
            IF (ipiv(k)>0) THEN
              CALL DGEMV('Transpose', k-1, nrhs, -one, b, ldb, a(1,k),      &
                1_ip_, one, b(k,1), ldb)
              kp = ipiv(k)
              IF (kp/=k) CALL DSWAP(nrhs, b(k,1), ldb, b(kp,1), ldb)
              k = k + 1
            ELSE
              CALL DGEMV('Transpose', k-1, nrhs, -one, b, ldb, a(1,k),      &
                1_ip_, one, b(k,1), ldb)
              CALL DGEMV('Transpose', k-1, nrhs, -one, b, ldb, a(1,k+1),    &
                1_ip_, one, b(k+1,1), ldb)
              kp = -ipiv(k)
              IF (kp/=k) CALL DSWAP(nrhs, b(k,1), ldb, b(kp,1), ldb)
              k = k + 2
            END IF
            GO TO 40
 50         CONTINUE
          ELSE
            k = 1
 60         CONTINUE
            IF (k>n) GO TO 80
            IF (ipiv(k)>0) THEN
              kp = ipiv(k)
              IF (kp/=k) CALL DSWAP(nrhs, b(k,1), ldb, b(kp,1), ldb)
              IF (k<n) CALL DGER(n-k, nrhs, -one, a(k+1,k), 1_ip_, b(k,1),  &
                ldb, b(k+1,1), ldb)
              CALL DSCAL(nrhs, one/a(k,k), b(k,1), ldb)
              k = k + 1
            ELSE
              kp = -ipiv(k)
              IF (kp/=k+1) CALL DSWAP(nrhs, b(k+1,1), ldb, b(kp,1), ldb)
              IF (k<n-1) THEN
                CALL DGER(n-k-1, nrhs, -one, a(k+2,k), 1_ip_, b(k,1), ldb,  &
                  b(k+2,1), ldb)
                CALL DGER(n-k-1, nrhs, -one, a(k+2,k+1), 1_ip_, b(k+1,1),   &
                  ldb, b(k+2,1), ldb)
              END IF
              akm1k = a(k+1, k)
              akm1 = a(k, k)/akm1k
              ak = a(k+1, k+1)/akm1k
              denom = akm1*ak - one
              DO j = 1, nrhs
                bkm1 = b(k, j)/akm1k
                bk = b(k+1, j)/akm1k
                b(k, j) = (ak*bkm1-bk)/denom
                b(k+1, j) = (akm1*bk-bkm1)/denom
              END DO
              k = k + 2
            END IF
            GO TO 60
 80         CONTINUE
            k = n
 90         CONTINUE
            IF (k<1) GO TO 100
            IF (ipiv(k)>0) THEN
              IF (k<n) CALL DGEMV('Transpose', n-k, nrhs, -one, b(k+1,1),   &
                ldb, a(k+1,k), 1_ip_, one, b(k,1), ldb)
              kp = ipiv(k)
              IF (kp/=k) CALL DSWAP(nrhs, b(k,1), ldb, b(kp,1), ldb)
              k = k - 1
            ELSE
              IF (k<n) THEN
                CALL DGEMV('Transpose', n-k, nrhs, -one, b(k+1,1), ldb,     &
                  a(k+1,k), 1_ip_, one, b(k,1), ldb)
                CALL DGEMV('Transpose', n-k, nrhs, -one, b(k+1,1), ldb,     &
                  a(k+1,k-1), 1_ip_, one, b(k-1,1), ldb)
              END IF
              kp = -ipiv(k)
              IF (kp/=k) CALL DSWAP(nrhs, b(k,1), ldb, b(kp,1), ldb)
              k = k - 2
            END IF
            GO TO 90
 100        CONTINUE
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DTREXC(compq, n, t, ldt, q, ldq, ifst, ilst, work, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: compq
          INTEGER(ip_) :: ifst, ilst, info, ldq, ldt, n
          REAL(rp_) :: q(ldq, *), t(ldt, *), work(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          LOGICAL :: wantq
          INTEGER(ip_) :: here, nbf, nbl, nbnext
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: DLAEXC, XERBLA2
          INTRINSIC :: MAX
          info = 0
          wantq = LSAME(compq, 'V')
          IF (.NOT. wantq .AND. .NOT. LSAME(compq,'N')) THEN
            info = -1
          ELSE IF (n<0) THEN
            info = -2
          ELSE IF (ldt<MAX(1,n)) THEN
            info = -4
          ELSE IF (ldq<1 .OR. (wantq .AND. ldq<MAX(1,n))) THEN
            info = -6
          ELSE IF ((ifst<1 .OR. ifst>n) .AND. (n>0)) THEN
            info = -7
          ELSE IF ((ilst<1 .OR. ilst>n) .AND. (n>0)) THEN
            info = -8
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('TREXC', -info)
            RETURN
          END IF
          IF (n<=1) RETURN
          IF (ifst>1) THEN
            IF (t(ifst,ifst-1)/=zero) ifst = ifst - 1
          END IF
          nbf = 1
          IF (ifst<n) THEN
            IF (t(ifst+1,ifst)/=zero) nbf = 2
          END IF
          IF (ilst>1) THEN
            IF (t(ilst,ilst-1)/=zero) ilst = ilst - 1
          END IF
          nbl = 1
          IF (ilst<n) THEN
            IF (t(ilst+1,ilst)/=zero) nbl = 2
          END IF
          IF (ifst==ilst) RETURN
          IF (ifst<ilst) THEN
            IF (nbf==2 .AND. nbl==1) ilst = ilst - 1
            IF (nbf==1 .AND. nbl==2) ilst = ilst + 1
            here = ifst
 10         CONTINUE
            IF (nbf==1 .OR. nbf==2) THEN
              nbnext = 1
              IF (here+nbf+1<=n) THEN
                IF (t(here+nbf+1,here+nbf)/=zero) nbnext = 2
              END IF
              CALL DLAEXC(wantq, n, t, ldt, q, ldq, here, nbf, nbnext,      &
                work, info)
              IF (info/=0) THEN
                ilst = here
                RETURN
              END IF
              here = here + nbnext
              IF (nbf==2) THEN
                IF (t(here+1,here)==zero) nbf = 3
              END IF
            ELSE
              nbnext = 1
              IF (here+3<=n) THEN
                IF (t(here+3,here+2)/=zero) nbnext = 2
              END IF
              CALL DLAEXC(wantq, n, t, ldt, q, ldq, here+1, 1_ip_, nbnext,  &
                work, info)
              IF (info/=0) THEN
                ilst = here
                RETURN
              END IF
              IF (nbnext==1) THEN
                CALL DLAEXC(wantq, n, t, ldt, q, ldq, here, 1_ip_, nbnext,  &
                  work, info)
                here = here + 1
              ELSE
                IF (t(here+2,here+1)==zero) nbnext = 1
                IF (nbnext==2) THEN
                  CALL DLAEXC(wantq, n, t, ldt, q, ldq, here, 1_ip_,        &
                    nbnext, work, info)
                  IF (info/=0) THEN
                    ilst = here
                    RETURN
                  END IF
                  here = here + 2
                ELSE
                  CALL DLAEXC(wantq, n, t, ldt, q, ldq, here, 1_ip_, 1_ip_, &
                    work, info)
                  CALL DLAEXC(wantq, n, t, ldt, q, ldq, here+1, 1_ip_,      &
                    1_ip_, work, info)
                  here = here + 2
                END IF
              END IF
            END IF
            IF (here<ilst) GO TO 10
          ELSE
            here = ifst
 20         CONTINUE
            IF (nbf==1 .OR. nbf==2) THEN
              nbnext = 1
              IF (here>=3) THEN
                IF (t(here-1,here-2)/=zero) nbnext = 2
              END IF
              CALL DLAEXC(wantq, n, t, ldt, q, ldq, here-nbnext, nbnext,    &
                nbf, work, info)
              IF (info/=0) THEN
                ilst = here
                RETURN
              END IF
              here = here - nbnext
              IF (nbf==2) THEN
                IF (t(here+1,here)==zero) nbf = 3
              END IF
            ELSE
              nbnext = 1
              IF (here>=3) THEN
                IF (t(here-1,here-2)/=zero) nbnext = 2
              END IF
              CALL DLAEXC(wantq, n, t, ldt, q, ldq, here-nbnext, nbnext,    &
                1_ip_, work, info)
              IF (info/=0) THEN
                ilst = here
                RETURN
              END IF
              IF (nbnext==1) THEN
                CALL DLAEXC(wantq, n, t, ldt, q, ldq, here, nbnext, 1_ip_,  &
                  work, info)
                here = here - 1
              ELSE
                IF (t(here,here-1)==zero) nbnext = 1
                IF (nbnext==2) THEN
                  CALL DLAEXC(wantq, n, t, ldt, q, ldq, here-1, 2_ip_,      &
                    1_ip_, work, info)
                  IF (info/=0) THEN
                    ilst = here
                    RETURN
                  END IF
                  here = here - 2
                ELSE
                  CALL DLAEXC(wantq, n, t, ldt, q, ldq, here, 1_ip_, 1_ip_, &
                    work, info)
                  CALL DLAEXC(wantq, n, t, ldt, q, ldq, here-1, 1_ip_,      &
                    1_ip_, work, info)
                  here = here - 2
                END IF
              END IF
            END IF
            IF (here>ilst) GO TO 20
          END IF
          ilst = here
          RETURN
        END SUBROUTINE

        SUBROUTINE DTRTRS(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: diag, trans, uplo
          INTEGER(ip_) :: info, lda, ldb, n, nrhs
          REAL(rp_) :: a(lda, *), b(ldb, *)
          REAL(rp_) :: zero, one
          PARAMETER (zero=0.0_rp_, one=1.0_rp_)
          LOGICAL :: nounit
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: DTRSM, XERBLA2
          INTRINSIC :: MAX
          info = 0
          nounit = LSAME(diag, 'N')
          IF (.NOT. LSAME(uplo,'U') .AND. .NOT. LSAME(uplo,'L')) THEN
            info = -1
          ELSE IF (.NOT. LSAME(trans,'N') .AND. .NOT. LSAME(trans,& 
            'T') .AND. .NOT. LSAME(trans,'C')) THEN
            info = -2
          ELSE IF (.NOT. nounit .AND. .NOT. LSAME(diag,'U')) THEN
            info = -3
          ELSE IF (n<0) THEN
            info = -4
          ELSE IF (nrhs<0) THEN
            info = -5
          ELSE IF (lda<MAX(1,n)) THEN
            info = -7
          ELSE IF (ldb<MAX(1,n)) THEN
            info = -9
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('TRTRS', -info)
            RETURN
          END IF
          IF (n==0) RETURN
          IF (nounit) THEN
            DO info = 1, n
              IF (a(info,info)==zero) RETURN
            END DO
          END IF
          info = 0
          CALL DTRSM('Left', uplo, trans, diag, n, nrhs, one, a, lda, b,    &
            ldb)
          RETURN
        END SUBROUTINE

        SUBROUTINE DTZRZF(m, n, a, lda, tau, work, lwork, info)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: info, lda, lwork, m, n
          REAL(rp_) :: a(lda, *), tau(*), work(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          LOGICAL :: lquery
          INTEGER(ip_) :: i, ib, iws, ki, kk, ldwork, lwkmin, lwkopt, m1,   &
            mu, nb, nbmin, nx
          EXTERNAL :: XERBLA2, DLARZB, DLARZT, DLATRZ
          INTRINSIC :: MAX, MIN
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: ILAENV2
          info = 0
          lquery = (lwork==-1)
          IF (m<0) THEN
            info = -1
          ELSE IF (n<m) THEN
            info = -2
          ELSE IF (lda<MAX(1,m)) THEN
            info = -4
          END IF
          IF (info==0) THEN
            IF (m==0 .OR. m==n) THEN
              lwkopt = 1
              lwkmin = 1
            ELSE
              nb = ILAENV2(1_ip_, 'GERQF', ' ', m, n, -1_ip_, -1_ip_)
              lwkopt = m*nb
              lwkmin = MAX(1, m)
            END IF
            work(1) = REAL(lwkopt,rp_)
            IF (lwork<lwkmin .AND. .NOT. lquery) THEN
              info = -7
            END IF
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('TZRZF', -info)
            RETURN
          ELSE IF (lquery) THEN
            RETURN
          END IF
          IF (m==0) THEN
            RETURN
          ELSE IF (m==n) THEN
            DO i = 1, n
              tau(i) = zero
            END DO
            RETURN
          END IF
          nbmin = 2
          nx = 1
          iws = m
          IF (nb>1 .AND. nb<m) THEN
            nx = MAX(0_ip_, ILAENV2(3_ip_,'GERQF',' ',m,n,-1_ip_,-1_ip_))
            IF (nx<m) THEN
              ldwork = m
              iws = ldwork*nb
              IF (lwork<iws) THEN
                nb = lwork/ldwork
                nbmin = MAX(2_ip_, ILAENV2(2_ip_,'GERQF',' ',m,n,-1_ip_,    &
                  -1_ip_))
              END IF
            END IF
          END IF
          IF (nb>=nbmin .AND. nb<m .AND. nx<m) THEN
            m1 = MIN(m+1, n)
            ki = ((m-nx-1)/nb)*nb
            kk = MIN(m, ki+nb)
            DO i = m - kk + ki + 1, m - kk + 1, -nb
              ib = MIN(m-i+1, nb)
              CALL DLATRZ(ib, n-i+1, n-m, a(i,i), lda, tau(i), work)
              IF (i>1) THEN
                CALL DLARZT('Backward', 'Rowwise', n-m, ib, a(i,m1), lda,   &
                  tau(i), work, ldwork)
                CALL DLARZB('Right', 'No transpose', 'Backward', 'Rowwise', &
                  i-1, n-i+1, ib, n-m, a(i,m1), lda, work, ldwork, a(1,i),  &
                  lda, work(ib+1), ldwork)
              END IF
            END DO
            mu = i + nb - 1
          ELSE
            mu = m
          END IF
          IF (mu>0) CALL DLATRZ(mu, n, n-m, a, lda, tau, work)
          work(1) = REAL(lwkopt,rp_)
          RETURN
        END SUBROUTINE

        INTEGER(ip_) FUNCTION ILADLC(m, n, a, lda)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: m, n, lda
          REAL(rp_) :: a(lda, *)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          INTEGER(ip_) :: i
          IF (n==0) THEN
            ILADLC = n
          ELSE IF (a(1,n)/=zero .OR. a(m,n)/=zero) THEN
            ILADLC = n
          ELSE
            DO ILADLC = n, 1_ip_, -1_ip_
              DO i = 1, m
                IF (a(i,ILADLC)/=zero) RETURN
              END DO
            END DO
          END IF
          RETURN
        END FUNCTION

        INTEGER(ip_) FUNCTION ILADLR(m, n, a, lda)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: m, n, lda
          REAL(rp_) :: a(lda, *)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          INTEGER(ip_) :: i, j
          IF (m==0) THEN
            ILADLR = m
          ELSE IF (a(m,1)/=zero .OR. a(m,n)/=zero) THEN
            ILADLR = m
          ELSE
            ILADLR = 0
            DO j = 1, n
              i = m
              DO WHILE ((a(MAX(i,1),j)==zero) .AND. (i>=1))
                i = i - 1
              END DO
              ILADLR = MAX(ILADLR, i)
            END DO
          END IF
          RETURN
        END FUNCTION

!  replacement ILAENV

        INTEGER(ip_) FUNCTION ILAENV(ispec, name, opts, n1, n2, n3, n4)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER (*) :: name, opts
          INTEGER(ip_) :: ispec, n1, n2, n3, n4
          INTEGER(ip_) :: len_name
          INTEGER(ip_) :: ILAENV2
          EXTERNAL :: ILAENV2
          len_name = LEN( name, ip_ )
          ILAENV = ILAENV2(ispec, name(2:len_name), opts, n1, n2, n3, n4)
          RETURN
        END FUNCTION

!  modified version of ILAENV for which the base_name excludes the
!  first (precision identifying) character

        INTEGER(ip_) FUNCTION ILAENV2(ispec, base_name, opts, n1, n2, n3, n4)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER (*) :: base_name, opts
          INTEGER(ip_) :: ispec, n1, n2, n3, n4
          INTEGER(ip_) :: i, ic, iz, nb, nbmin, nx
          LOGICAL :: cname, sname, twostage
          CHARACTER :: c1*1, c2*2, c4*2, c3*3, subnam*16
          INTRINSIC :: CHAR, ICHAR, INT, MIN, REAL
          INTEGER(ip_) :: IEEECK, IPARMQ, IPARAM2STAGE
          EXTERNAL :: IEEECK, IPARMQ, IPARAM2STAGE
!  added for galahad templating
          CHARACTER ( LEN = LEN( base_name ) + 4 ) :: name
          CHARACTER ( LEN = 1 ) :: real_prefix
          CHARACTER ( LEN = 3 ) :: int_suffix
          IF ( rp_ == REAL32 ) THEN
            real_prefix = ''
          ELSE IF ( rp_ == REAL64 ) THEN
            real_prefix = 'D'
          ELSE
            real_prefix = 'Q'
          END IF
          IF ( ip_ == INT64 ) THEN
            int_suffix = '_64'
          ELSE
            int_suffix = '   '
          END IF
          name = real_prefix // base_name // int_suffix

          GO TO (10, 10, 10, 80, 90, 100, 110, 120, 130, 140, 150, 160,     &
            160, 160, 160, 160) ispec
          ILAENV2 = -1
          RETURN
 10       CONTINUE
          ILAENV2 = 1
          subnam = name
          ic = ICHAR(subnam(1:1))
          iz = ICHAR('Z')
          IF (iz==90 .OR. iz==122) THEN
            IF (ic>=97 .AND. ic<=122) THEN
              subnam(1:1) = CHAR(ic-32)
              DO i = 2, 6
                ic = ICHAR(subnam(i:i))
                IF (ic>=97 .AND. ic<=122) subnam(i:i) = CHAR(ic-32)
              END DO
            END IF
          ELSE IF (iz==233 .OR. iz==169) THEN
            IF ((ic>=129 .AND. ic<=137) .OR. (ic>=145 .AND. ic<=153) .OR.   &
              (ic>=162 .AND. ic<=169)) THEN
              subnam(1:1) = CHAR(ic+64)
              DO i = 2, 6
                ic = ICHAR(subnam(i:i))
                IF ((ic>=129 .AND. ic<=137) .OR. (ic>=145 .AND. ic<=153)    &
                  .OR. (ic>=162 .AND. ic<=169)) subnam(i:i) = CHAR(ic+64)
              END DO
            END IF
          ELSE IF (iz==218 .OR. iz==250) THEN
            IF (ic>=225 .AND. ic<=250) THEN
              subnam(1:1) = CHAR(ic-32)
              DO i = 2, 6
                ic = ICHAR(subnam(i:i))
                IF (ic>=225 .AND. ic<=250) subnam(i:i) = CHAR(ic-32)
              END DO
            END IF
          END IF
          c1 = subnam(1:1)
! update for quadruple precision
          sname = c1 == 'S' .OR. c1 == 'D' .OR. c1 == 'Q'
          cname = c1 == 'C' .OR. c1 == 'Z' .OR. c1 == 'X'
          IF (.NOT. (cname .OR. sname)) RETURN
          c2 = subnam(2:3)
          c3 = subnam(4:6)
          c4 = c3(2:3)
          twostage = LEN(subnam) >= 11 .AND. subnam(11:11) == '2'
          GO TO (50, 60, 70) ispec
 50       CONTINUE
          nb = 1
          IF (subnam(2:6)=='LAORH') THEN
            IF (sname) THEN
              nb = 32
            ELSE
              nb = 32
            END IF
          ELSE IF (c2=='GE') THEN
            IF (c3=='TRF') THEN
              IF (sname) THEN
                nb = 64
              ELSE
                nb = 64
              END IF
            ELSE IF (c3=='QRF' .OR. c3=='RQF' .OR. c3=='LQF' .OR.           &
              c3=='QLF') THEN
              IF (sname) THEN
                nb = 32
              ELSE
                nb = 32
              END IF
            ELSE IF (c3=='QR ') THEN
              IF (n3==1) THEN
                IF (sname) THEN
                  IF ((n1*n2<=131072) .OR. (n1<=8192)) THEN
                    nb = n1
                  ELSE
                    nb = 32768/n2
                  END IF
                ELSE
                  IF ((n1*n2<=131072) .OR. (n1<=8192)) THEN
                    nb = n1
                  ELSE
                    nb = 32768/n2
                  END IF
                END IF
              ELSE
                IF (sname) THEN
                  nb = 1
                ELSE
                  nb = 1
                END IF
              END IF
            ELSE IF (c3=='LQ ') THEN
              IF (n3==2) THEN
                IF (sname) THEN
                  IF ((n1*n2<=131072) .OR. (n1<=8192)) THEN
                    nb = n1
                  ELSE
                    nb = 32768/n2
                  END IF
                ELSE
                  IF ((n1*n2<=131072) .OR. (n1<=8192)) THEN
                    nb = n1
                  ELSE
                    nb = 32768/n2
                  END IF
                END IF
              ELSE
                IF (sname) THEN
                  nb = 1
                ELSE
                  nb = 1
                END IF
              END IF
            ELSE IF (c3=='HRD') THEN
              IF (sname) THEN
                nb = 32
              ELSE
                nb = 32
              END IF
            ELSE IF (c3=='BRD') THEN
              IF (sname) THEN
                nb = 32
              ELSE
                nb = 32
              END IF
            ELSE IF (c3=='TRI') THEN
              IF (sname) THEN
                nb = 64
              ELSE
                nb = 64
              END IF
            END IF
          ELSE IF (c2=='PO') THEN
            IF (c3=='TRF') THEN
              IF (sname) THEN
                nb = 64
              ELSE
                nb = 64
              END IF
            END IF
          ELSE IF (c2=='SY') THEN
            IF (c3=='TRF') THEN
              IF (sname) THEN
                IF (twostage) THEN
                  nb = 192
                ELSE
                  nb = 64
                END IF
              ELSE
                IF (twostage) THEN
                  nb = 192
                ELSE
                  nb = 64
                END IF
              END IF
            ELSE IF (sname .AND. c3=='TRD') THEN
              nb = 32
            ELSE IF (sname .AND. c3=='GST') THEN
              nb = 64
            END IF
          ELSE IF (cname .AND. c2=='HE') THEN
            IF (c3=='TRF') THEN
              IF (twostage) THEN
                nb = 192
              ELSE
                nb = 64
              END IF
            ELSE IF (c3=='TRD') THEN
              nb = 32
            ELSE IF (c3=='GST') THEN
              nb = 64
            END IF
          ELSE IF (sname .AND. c2=='OR') THEN
            IF (c3(1:1)=='G') THEN
              IF (c4=='QR' .OR. c4=='RQ' .OR. c4=='LQ' .OR. c4=='QL' .OR.   &
                c4=='HR' .OR. c4=='TR' .OR. c4=='BR') THEN
                nb = 32
              END IF
            ELSE IF (c3(1:1)=='M') THEN
              IF (c4=='QR' .OR. c4=='RQ' .OR. c4=='LQ' .OR. c4=='QL' .OR.   &
                c4=='HR' .OR. c4=='TR' .OR. c4=='BR') THEN
                nb = 32
              END IF
            END IF
          ELSE IF (cname .AND. c2=='UN') THEN
            IF (c3(1:1)=='G') THEN
              IF (c4=='QR' .OR. c4=='RQ' .OR. c4=='LQ' .OR. c4=='QL' .OR.   &
                c4=='HR' .OR. c4=='TR' .OR. c4=='BR') THEN
                nb = 32
              END IF
            ELSE IF (c3(1:1)=='M') THEN
              IF (c4=='QR' .OR. c4=='RQ' .OR. c4=='LQ' .OR. c4=='QL' .OR.   &
                c4=='HR' .OR. c4=='TR' .OR. c4=='BR') THEN
                nb = 32
              END IF
            END IF
          ELSE IF (c2=='GB') THEN
            IF (c3=='TRF') THEN
              IF (sname) THEN
                IF (n4<=64) THEN
                  nb = 1
                ELSE
                  nb = 32
                END IF
              ELSE
                IF (n4<=64) THEN
                  nb = 1
                ELSE
                  nb = 32
                END IF
              END IF
            END IF
          ELSE IF (c2=='PB') THEN
            IF (c3=='TRF') THEN
              IF (sname) THEN
                IF (n2<=64) THEN
                  nb = 1
                ELSE
                  nb = 32
                END IF
              ELSE
                IF (n2<=64) THEN
                  nb = 1
                ELSE
                  nb = 32
                END IF
              END IF
            END IF
          ELSE IF (c2=='TR') THEN
            IF (c3=='TRI') THEN
              IF (sname) THEN
                nb = 64
              ELSE
                nb = 64
              END IF
            ELSE IF (c3=='EVC') THEN
              IF (sname) THEN
                nb = 64
              ELSE
                nb = 64
              END IF
            END IF
          ELSE IF (c2=='LA') THEN
            IF (c3=='UUM') THEN
              IF (sname) THEN
                nb = 64
              ELSE
                nb = 64
              END IF
            END IF
          ELSE IF (sname .AND. c2=='ST') THEN
            IF (c3=='EBZ') THEN
              nb = 1
            END IF
          ELSE IF (c2=='GG') THEN
            nb = 32
            IF (c3=='HD3') THEN
              IF (sname) THEN
                nb = 32
              ELSE
                nb = 32
              END IF
            END IF
          END IF
          ILAENV2 = nb
          RETURN
 60       CONTINUE
          nbmin = 2
          IF (c2=='GE') THEN
            IF (c3=='QRF' .OR. c3=='RQF' .OR. c3=='LQF' .OR. c3=='QLF')     &
              THEN
              IF (sname) THEN
                nbmin = 2
              ELSE
                nbmin = 2
              END IF
            ELSE IF (c3=='HRD') THEN
              IF (sname) THEN
                nbmin = 2
              ELSE
                nbmin = 2
              END IF
            ELSE IF (c3=='BRD') THEN
              IF (sname) THEN
                nbmin = 2
              ELSE
                nbmin = 2
              END IF
            ELSE IF (c3=='TRI') THEN
              IF (sname) THEN
                nbmin = 2
              ELSE
                nbmin = 2
              END IF
            END IF
          ELSE IF (c2=='SY') THEN
            IF (c3=='TRF') THEN
              IF (sname) THEN
                nbmin = 8
              ELSE
                nbmin = 8
              END IF
            ELSE IF (sname .AND. c3=='TRD') THEN
              nbmin = 2
            END IF
          ELSE IF (cname .AND. c2=='HE') THEN
            IF (c3=='TRD') THEN
              nbmin = 2
            END IF
          ELSE IF (sname .AND. c2=='OR') THEN
            IF (c3(1:1)=='G') THEN
              IF (c4=='QR' .OR. c4=='RQ' .OR. c4=='LQ' .OR. c4=='QL' .OR.   &
                c4=='HR' .OR. c4=='TR' .OR. c4=='BR') THEN
                nbmin = 2
              END IF
            ELSE IF (c3(1:1)=='M') THEN
              IF (c4=='QR' .OR. c4=='RQ' .OR. c4=='LQ' .OR. c4=='QL' .OR.   &
                c4=='HR' .OR. c4=='TR' .OR. c4=='BR') THEN
                nbmin = 2
              END IF
            END IF
          ELSE IF (cname .AND. c2=='UN') THEN
            IF (c3(1:1)=='G') THEN
              IF (c4=='QR' .OR. c4=='RQ' .OR. c4=='LQ' .OR. c4=='QL' .OR.   &
                c4=='HR' .OR. c4=='TR' .OR. c4=='BR') THEN
                nbmin = 2
              END IF
            ELSE IF (c3(1:1)=='M') THEN
              IF (c4=='QR' .OR. c4=='RQ' .OR. c4=='LQ' .OR. c4=='QL' .OR.   &
                c4=='HR' .OR. c4=='TR' .OR. c4=='BR') THEN
                nbmin = 2
              END IF
            END IF
          ELSE IF (c2=='GG') THEN
            nbmin = 2
            IF (c3=='HD3') THEN
              nbmin = 2
            END IF
          END IF
          ILAENV2 = nbmin
          RETURN
 70       CONTINUE
          nx = 0
          IF (c2=='GE') THEN
            IF (c3=='QRF' .OR. c3=='RQF' .OR. c3=='LQF' .OR. c3=='QLF')     &
              THEN
              IF (sname) THEN
                nx = 128
              ELSE
                nx = 128
              END IF
            ELSE IF (c3=='HRD') THEN
              IF (sname) THEN
                nx = 128
              ELSE
                nx = 128
              END IF
            ELSE IF (c3=='BRD') THEN
              IF (sname) THEN
                nx = 128
              ELSE
                nx = 128
              END IF
            END IF
          ELSE IF (c2=='SY') THEN
            IF (sname .AND. c3=='TRD') THEN
              nx = 32
            END IF
          ELSE IF (cname .AND. c2=='HE') THEN
            IF (c3=='TRD') THEN
              nx = 32
            END IF
          ELSE IF (sname .AND. c2=='OR') THEN
            IF (c3(1:1)=='G') THEN
              IF (c4=='QR' .OR. c4=='RQ' .OR. c4=='LQ' .OR. c4=='QL' .OR.   &
                c4=='HR' .OR. c4=='TR' .OR. c4=='BR') THEN
                nx = 128
              END IF
            END IF
          ELSE IF (cname .AND. c2=='UN') THEN
            IF (c3(1:1)=='G') THEN
              IF (c4=='QR' .OR. c4=='RQ' .OR. c4=='LQ' .OR. c4=='QL' .OR.   &
                c4=='HR' .OR. c4=='TR' .OR. c4=='BR') THEN
                nx = 128
              END IF
            END IF
          ELSE IF (c2=='GG') THEN
            nx = 128
            IF (c3=='HD3') THEN
              nx = 128
            END IF
          END IF
          ILAENV2 = nx
          RETURN
 80       CONTINUE
          ILAENV2 = 6
          RETURN
 90       CONTINUE
          ILAENV2 = 2
          RETURN
 100      CONTINUE
          ILAENV2 = INT(REAL(MIN(n1,n2))*1.6E0)
          RETURN
 110      CONTINUE
          ILAENV2 = 1
          RETURN
 120      CONTINUE
          ILAENV2 = 50
          RETURN
 130      CONTINUE
          ILAENV2 = 25
          RETURN
 140      CONTINUE
          ILAENV2 = 1
          IF (ILAENV2==1) THEN
            ILAENV2 = IEEECK(1_ip_, 0.0, 1.0)
          END IF
          RETURN
 150      CONTINUE
          ILAENV2 = 1
          IF (ILAENV2==1) THEN
            ILAENV2 = IEEECK(0_ip_, 0.0, 1.0)
          END IF
          RETURN
 160      CONTINUE
          ILAENV2 = IPARMQ(ispec, TRIM(name), opts, n1, n2, n3, n4)
          RETURN
        END FUNCTION

        INTEGER(ip_) FUNCTION IPARMQ(ispec, name, opts, n, ilo, ihi, lwork)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: ihi, ilo, ispec, lwork, n
          CHARACTER :: name*(*), opts*(*)
          INTEGER(ip_) :: inmin, inwin, inibl, ishfts, iacc22
          PARAMETER (inmin=12, inwin=13, inibl=14, ishfts=15, iacc22=16)
          INTEGER(ip_) :: nmin, k22min, kacmin, nibble, knwswp
          PARAMETER (nmin=75, k22min=14, kacmin=14, nibble=14, knwswp=500)
          REAL(r4_) :: two
          PARAMETER (two=2.0)
          INTEGER(ip_) :: nh, ns
          INTEGER(ip_) :: i, ic, iz
          CHARACTER :: subnam*6
          INTRINSIC :: LOG, MAX, MOD, NINT, REAL
          IF ((ispec==ishfts) .OR. (ispec==inwin) .OR. (ispec==iacc22))     &
            THEN
            nh = ihi - ilo + 1
            ns = 2
            IF (nh>=30) ns = 4
            IF (nh>=60) ns = 10
            IF (nh>=150) ns = MAX(10, nh/NINT(LOG(REAL(nh))/LOG(two)))
            IF (nh>=590) ns = 64
            IF (nh>=3000) ns = 128
            IF (nh>=6000) ns = 256
            ns = MAX( 2_ip_, ns-MOD(ns,2))
          END IF
          IF (ispec==inmin) THEN
            IPARMQ = nmin
          ELSE IF (ispec==inibl) THEN
            IPARMQ = nibble
          ELSE IF (ispec==ishfts) THEN
            IPARMQ = ns
          ELSE IF (ispec==inwin) THEN
            IF (nh<=knwswp) THEN
              IPARMQ = ns
            ELSE
              IPARMQ = 3*ns/2
            END IF
          ELSE IF (ispec==iacc22) THEN
            IPARMQ = 0
            subnam = name
            ic = ICHAR(subnam(1:1))
            iz = ICHAR('Z')
            IF (iz==90 .OR. iz==122) THEN
              IF (ic>=97 .AND. ic<=122) THEN
                subnam(1:1) = CHAR(ic-32)
                DO i = 2, 6
                  ic = ICHAR(subnam(i:i))
                  IF (ic>=97 .AND. ic<=122) subnam(i:i) = CHAR(ic-32)
                END DO
              END IF
            ELSE IF (iz==233 .OR. iz==169) THEN
              IF ((ic>=129 .AND. ic<=137) .OR. (ic>=145 .AND. ic<=153)      &
                .OR. (ic>=162 .AND. ic<=169)) THEN
                subnam(1:1) = CHAR(ic+64)
                DO i = 2, 6
                  ic = ICHAR(subnam(i:i))
                  IF ((ic>=129 .AND. ic<=137) .OR. (ic>=145 .AND. ic<=153)  &
                    .OR. (ic>=162 .AND. ic<=169)) subnam(i:i) = CHAR(ic+64)
                END DO
              END IF
            ELSE IF (iz==218 .OR. iz==250) THEN
              IF (ic>=225 .AND. ic<=250) THEN
                subnam(1:1) = CHAR(ic-32)
                DO i = 2, 6
                  ic = ICHAR(subnam(i:i))
                  IF (ic>=225 .AND. ic<=250) subnam(i:i) = CHAR(ic-32)
                END DO
              END IF
            END IF
            IF (subnam(2:6)=='GGHRD' .OR. subnam(2:6)=='GGHD3') THEN
              IPARMQ = 1
              IF (nh>=k22min) IPARMQ = 2
            ELSE IF (subnam(4:6)=='EXC') THEN
              IF (nh>=kacmin) IPARMQ = 1
              IF (nh>=k22min) IPARMQ = 2
            ELSE IF (subnam(2:6)=='HSEQR' .OR. subnam(2:5)=='LAQR') THEN
              IF (ns>=kacmin) IPARMQ = 1
              IF (ns>=k22min) IPARMQ = 2
            END IF
          ELSE
            IPARMQ = -1
          END IF
        END FUNCTION

        INTEGER(ip_) FUNCTION IEEECK(ispec, zero, one)
         USE BLAS_LAPACK_KINDS_precision
         INTEGER(ip_) :: ispec
         REAL :: one, zero
         REAL :: nan1, nan2, nan3, nan4, nan5, nan6, neginf, negzro,      &
           newzro, posinf
#ifdef NO_IEEECK
         IEEECK = 0
#else
         IEEECK = 1
         posinf = one/zero
         IF (posinf<=one) THEN
           IEEECK = 0
           RETURN
         END IF
         neginf = -one/zero
         IF (neginf>=zero) THEN
           IEEECK = 0
           RETURN
         END IF
         negzro = one/(neginf+one)
         IF (negzro/=zero) THEN
           IEEECK = 0
           RETURN
         END IF
         neginf = one/negzro
         IF (neginf>=zero) THEN
           IEEECK = 0
           RETURN
         END IF
         newzro = negzro + zero
         IF (newzro/=zero) THEN
           IEEECK = 0
           RETURN
         END IF
         posinf = one/newzro
         IF (posinf<=one) THEN
           IEEECK = 0
           RETURN
         END IF
         neginf = neginf*posinf
         IF (neginf>=zero) THEN
           IEEECK = 0
           RETURN
         END IF
         posinf = posinf*posinf
         IF (posinf<=one) THEN
           IEEECK = 0
           RETURN
         END IF
         IF (ispec==0) RETURN
         nan1 = posinf + neginf
         nan2 = posinf/neginf
         nan3 = posinf/posinf
         nan4 = posinf*zero
         nan5 = neginf*negzro
         nan6 = nan5*zero
         IF (nan1==nan1) THEN
           IEEECK = 0
           RETURN
         END IF
         IF (nan2==nan2) THEN
           IEEECK = 0
           RETURN
         END IF
         IF (nan3==nan3) THEN
           IEEECK = 0
           RETURN
         END IF
         IF (nan4==nan4) THEN
           IEEECK = 0
           RETURN
         END IF
         IF (nan5==nan5) THEN
           IEEECK = 0
           RETURN
         END IF
         IF (nan6==nan6) THEN
           IEEECK = 0
           RETURN
         END IF
#endif
         RETURN
       END FUNCTION
