! THIS VERSION: GALAHAD 5.1 - 2024-11-30 AT 10:00 GMT

#include "galahad_blas.h"

! Reference blas, see http://www.netlib.org/lapack/explore-html/

! C preprocessor will change, as appropriate,
!  ip_ to INT32 or INT64, 
!  rp_ to REAL32, REAL64 or REAL128
!  subroutine and function names starting with D to start with S or Q

        MODULE BLAS_LAPACK_KINDS_precision
          USE ISO_FORTRAN_ENV
          INTEGER, PARAMETER :: r4_ = REAL32
#ifdef INTEGER_64
          INTEGER, PARAMETER :: ip_ = INT64
#else
          INTEGER, PARAMETER :: ip_ = INT32
#endif
#ifdef REAL_32
          INTEGER, PARAMETER :: rp_ = REAL32
#elif REAL_128
          INTEGER, PARAMETER :: rp_ = REAL128
#else
          INTEGER, PARAMETER :: rp_ = REAL64
#endif
        END MODULE BLAS_LAPACK_KINDS_precision

        REAL(rp_) FUNCTION DASUM(n, dx, incx)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: incx, n
          REAL(rp_) :: dx(*)
          REAL(rp_) :: dtemp
          INTEGER(ip_) :: i, m, mp1, nincx
          INTRINSIC :: ABS, MOD
          DASUM = 0.0_rp_
          dtemp = 0.0_rp_
          IF (n<=0 .OR. incx<=0) RETURN
          IF (incx==1) THEN
            m = MOD(n, 6)
            IF (m/=0) THEN
              DO i = 1, m
                dtemp = dtemp + ABS(dx(i))
              END DO
              IF (n<6) THEN
                DASUM = dtemp
                RETURN
              END IF
            END IF
            mp1 = m + 1
            DO i = mp1, n, 6
              dtemp = dtemp + ABS(dx(i)) + ABS(dx(i+1)) + ABS(dx(i+2))   &
                + ABS(dx(i+3)) + ABS(dx(i+4)) + ABS(dx(i+5))
            END DO
          ELSE
            nincx = n*incx
            DO i = 1, nincx, incx
              dtemp = dtemp + ABS(dx(i))
            END DO
          END IF
          DASUM = dtemp
          RETURN
        END FUNCTION

        SUBROUTINE DAXPY(n, da, dx, incx, dy, incy)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: da
          INTEGER(ip_) :: incx, incy, n
          REAL(rp_) :: dx(*), dy(*)
          INTEGER(ip_) :: i, ix, iy, m, mp1
          INTRINSIC :: MOD
          IF (n<=0) RETURN
          IF (da==0.0_rp_) RETURN
          IF (incx==1 .AND. incy==1) THEN
            m = MOD(n, 4_ip_)
            IF (m/=0) THEN
              DO i = 1, m
                dy(i) = dy(i) + da*dx(i)
              END DO
            END IF
            IF (n<4) RETURN
            mp1 = m + 1
            DO i = mp1, n, 4
              dy(i) = dy(i) + da*dx(i)
              dy(i+1) = dy(i+1) + da*dx(i+1)
              dy(i+2) = dy(i+2) + da*dx(i+2)
              dy(i+3) = dy(i+3) + da*dx(i+3)
            END DO
          ELSE
            ix = 1
            iy = 1
            IF (incx<0) ix = (-n+1)*incx + 1
            IF (incy<0) iy = (-n+1)*incy + 1
            DO i = 1, n
              dy(iy) = dy(iy) + da*dx(ix)
              ix = ix + incx
              iy = iy + incy
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DCOPY(n, dx, incx, dy, incy)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: incx, incy, n
          REAL(rp_) :: dx(*), dy(*)
          INTEGER(ip_) :: i, ix, iy, m, mp1
          INTRINSIC :: MOD
          IF (n<=0) RETURN
          IF (incx==1 .AND. incy==1) THEN
            m = MOD(n, 7)
            IF (m/=0) THEN
              DO i = 1, m
                dy(i) = dx(i)
              END DO
              IF (n<7) RETURN
            END IF
            mp1 = m + 1
            DO i = mp1, n, 7
              dy(i) = dx(i)
              dy(i+1) = dx(i+1)
              dy(i+2) = dx(i+2)
              dy(i+3) = dx(i+3)
              dy(i+4) = dx(i+4)
              dy(i+5) = dx(i+5)
              dy(i+6) = dx(i+6)
            END DO
          ELSE
            ix = 1
            iy = 1
            IF (incx<0) ix = (-n+1)*incx + 1
            IF (incy<0) iy = (-n+1)*incy + 1
            DO i = 1, n
              dy(iy) = dx(ix)
              ix = ix + incx
              iy = iy + incy
            END DO
          END IF
          RETURN
        END SUBROUTINE

        REAL(rp_) FUNCTION DDOT(n, dx, incx, dy, incy)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: incx, incy, n
          REAL(rp_) :: dx(*), dy(*)
          REAL(rp_) :: dtemp
          INTEGER(ip_) :: i, ix, iy, m, mp1
          INTRINSIC :: MOD
          DDOT = 0.0_rp_
          dtemp = 0.0_rp_
          IF (n<=0) RETURN
          IF (incx==1 .AND. incy==1) THEN
            m = MOD(n, 5)
            IF (m/=0) THEN
              DO i = 1, m
                dtemp = dtemp + dx(i)*dy(i)
              END DO
              IF (n<5) THEN
                DDOT = dtemp
                RETURN
              END IF
            END IF
            mp1 = m + 1
            DO i = mp1, n, 5
              dtemp = dtemp + dx(i)*dy(i) + dx(i+1)*dy(i+1) +               &
                dx(i+2)*dy(i+2) + dx(i+3)*dy(i+3) + dx(i+4)*dy(i+4)
            END DO
          ELSE
            ix = 1
            iy = 1
            IF (incx<0) ix = (-n+1)*incx + 1
            IF (incy<0) iy = (-n+1)*incy + 1
            DO i = 1, n
              dtemp = dtemp + dx(ix)*dy(iy)
              ix = ix + incx
              iy = iy + incy
            END DO
          END IF
          DDOT = dtemp
          RETURN
        END FUNCTION

        SUBROUTINE DGEMM(transa, transb, m, n, k, alpha, a, lda, b, ldb,    &
          beta, c, ldc)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: alpha, beta
          INTEGER(ip_) :: k, lda, ldb, ldc, m, n
          CHARACTER :: transa, transb
          REAL(rp_) :: a(lda, *), b(ldb, *), c(ldc, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          INTRINSIC :: MAX
          REAL(rp_) :: temp
          INTEGER(ip_) :: i, info, j, l, nrowa, nrowb
          LOGICAL :: nota, notb
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          nota = LSAME(transa, 'N')
          notb = LSAME(transb, 'N')
          IF (nota) THEN
            nrowa = m
          ELSE
            nrowa = k
          END IF
          IF (notb) THEN
            nrowb = k
          ELSE
            nrowb = n
          END IF
          info = 0
          IF ((.NOT. nota) .AND. (.NOT. LSAME(transa, 'C')) .AND. &
            (.NOT. LSAME(transa,'T'))) THEN
            info = 1
          ELSE IF ((.NOT. notb) .AND. (.NOT. LSAME(transb, 'C')) .AND. &
            (.NOT. LSAME(transb,'T'))) THEN
            info = 2
          ELSE IF (m<0) THEN
            info = 3
          ELSE IF (n<0) THEN
            info = 4
          ELSE IF (k<0) THEN
            info = 5
          ELSE IF (lda<MAX(1,nrowa)) THEN
            info = 8
          ELSE IF (ldb<MAX(1,nrowb)) THEN
            info = 10
          ELSE IF (ldc<MAX(1,m)) THEN
            info = 13
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GEMM', info)
            RETURN
          END IF
          IF ((m==0) .OR. (n==0) .OR. (((alpha==zero) .OR. (k==0)) .AND. (  &
            beta==one))) RETURN
          IF (alpha==zero) THEN
            IF (beta==zero) THEN
              DO j = 1, n
                DO i = 1, m
                  c(i, j) = zero
                END DO
              END DO
            ELSE
              DO j = 1, n
                DO i = 1, m
                  c(i, j) = beta*c(i, j)
                END DO
              END DO
            END IF
            RETURN
          END IF
          IF (notb) THEN
            IF (nota) THEN
              DO j = 1, n
                IF (beta==zero) THEN
                  DO i = 1, m
                    c(i, j) = zero
                  END DO
                ELSE IF (beta/=one) THEN
                  DO i = 1, m
                    c(i, j) = beta*c(i, j)
                  END DO
                END IF
                DO l = 1, k
                  temp = alpha*b(l, j)
                  DO i = 1, m
                    c(i, j) = c(i, j) + temp*a(i, l)
                  END DO
                END DO
              END DO
            ELSE
              DO j = 1, n
                DO i = 1, m
                  temp = zero
                  DO l = 1, k
                    temp = temp + a(l, i)*b(l, j)
                  END DO
                  IF (beta==zero) THEN
                    c(i, j) = alpha*temp
                  ELSE
                    c(i, j) = alpha*temp + beta*c(i, j)
                  END IF
                END DO
              END DO
            END IF
          ELSE
            IF (nota) THEN
              DO j = 1, n
                IF (beta==zero) THEN
                  DO i = 1, m
                    c(i, j) = zero
                  END DO
                ELSE IF (beta/=one) THEN
                  DO i = 1, m
                    c(i, j) = beta*c(i, j)
                  END DO
                END IF
                DO l = 1, k
                  temp = alpha*b(j, l)
                  DO i = 1, m
                    c(i, j) = c(i, j) + temp*a(i, l)
                  END DO
                END DO
              END DO
            ELSE
              DO j = 1, n
                DO i = 1, m
                  temp = zero
                  DO l = 1, k
                    temp = temp + a(l, i)*b(j, l)
                  END DO
                  IF (beta==zero) THEN
                    c(i, j) = alpha*temp
                  ELSE
                    c(i, j) = alpha*temp + beta*c(i, j)
                  END IF
                END DO
              END DO
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DGEMV(trans, m, n, alpha, a, lda, x, incx, beta, y,      &
          incy)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: alpha, beta
          INTEGER(ip_) :: incx, incy, lda, m, n
          CHARACTER :: trans
          REAL(rp_) :: a(lda, *), x(*), y(*)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          REAL(rp_) :: temp
          INTEGER(ip_) :: i, info, ix, iy, j, jx, jy, kx, ky, lenx, leny
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          INTRINSIC :: MAX
          info = 0
          IF (.NOT. LSAME(trans,'N') .AND. .NOT. LSAME(trans,'T') &
            .AND. .NOT. LSAME(trans,'C')) THEN
            info = 1
          ELSE IF (m<0) THEN
            info = 2
          ELSE IF (n<0) THEN
            info = 3
          ELSE IF (lda<MAX(1,m)) THEN
            info = 6
          ELSE IF (incx==0) THEN
            info = 8
          ELSE IF (incy==0) THEN
            info = 11
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GEMV', info)
            RETURN
          END IF
          IF ((m==0) .OR. (n==0) .OR. ((alpha==zero) .AND. (beta== one)))   &
            RETURN
          IF (LSAME(trans,'N')) THEN
            lenx = n
            leny = m
          ELSE
            lenx = m
            leny = n
          END IF
          IF (incx>0) THEN
            kx = 1
          ELSE
            kx = 1 - (lenx-1)*incx
          END IF
          IF (incy>0) THEN
            ky = 1
          ELSE
            ky = 1 - (leny-1)*incy
          END IF
          IF (beta/=one) THEN
            IF (incy==1) THEN
              IF (beta==zero) THEN
                DO i = 1, leny
                  y(i) = zero
                END DO
              ELSE
                DO i = 1, leny
                  y(i) = beta*y(i)
                END DO
              END IF
            ELSE
              iy = ky
              IF (beta==zero) THEN
                DO i = 1, leny
                  y(iy) = zero
                  iy = iy + incy
                END DO
              ELSE
                DO i = 1, leny
                  y(iy) = beta*y(iy)
                  iy = iy + incy
                END DO
              END IF
            END IF
          END IF
          IF (alpha==zero) RETURN
          IF (LSAME(trans,'N')) THEN
            jx = kx
            IF (incy==1) THEN
              DO j = 1, n
                temp = alpha*x(jx)
                DO i = 1, m
                  y(i) = y(i) + temp*a(i, j)
                END DO
                jx = jx + incx
              END DO
            ELSE
              DO j = 1, n
                temp = alpha*x(jx)
                iy = ky
                DO i = 1, m
                  y(iy) = y(iy) + temp*a(i, j)
                  iy = iy + incy
                END DO
                jx = jx + incx
              END DO
            END IF
          ELSE
            jy = ky
            IF (incx==1) THEN
              DO j = 1, n
                temp = zero
                DO i = 1, m
                  temp = temp + a(i, j)*x(i)
                END DO
                y(jy) = y(jy) + alpha*temp
                jy = jy + incy
              END DO
            ELSE
              DO j = 1, n
                temp = zero
                ix = kx
                DO i = 1, m
                  temp = temp + a(i, j)*x(ix)
                  ix = ix + incx
                END DO
                y(jy) = y(jy) + alpha*temp
                jy = jy + incy
              END DO
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DGER(m, n, alpha, x, incx, y, incy, a, lda)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: alpha
          INTEGER(ip_) :: incx, incy, lda, m, n
          REAL(rp_) :: a(lda, *), x(*), y(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          REAL(rp_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jy, kx
          EXTERNAL :: XERBLA2
          INTRINSIC :: MAX
          info = 0
          IF (m<0) THEN
            info = 1
          ELSE IF (n<0) THEN
            info = 2
          ELSE IF (incx==0) THEN
            info = 5
          ELSE IF (incy==0) THEN
            info = 7
          ELSE IF (lda<MAX(1,m)) THEN
            info = 9
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('GER', info)
            RETURN
          END IF
          IF ((m==0) .OR. (n==0) .OR. (alpha==zero)) RETURN
          IF (incy>0) THEN
            jy = 1
          ELSE
            jy = 1 - (n-1)*incy
          END IF
          IF (incx==1) THEN
            DO j = 1, n
              IF (y(jy)/=zero) THEN
                temp = alpha*y(jy)
                DO i = 1, m
                  a(i, j) = a(i, j) + x(i)*temp
                END DO
              END IF
              jy = jy + incy
            END DO
          ELSE
            IF (incx>0) THEN
              kx = 1
            ELSE
              kx = 1 - (m-1)*incx
            END IF
            DO j = 1, n
              IF (y(jy)/=zero) THEN
                temp = alpha*y(jy)
                ix = kx
                DO i = 1, m
                  a(i, j) = a(i, j) + x(ix)*temp
                  ix = ix + incx
                END DO
              END IF
              jy = jy + incy
            END DO
          END IF
          RETURN
        END SUBROUTINE

        REAL(rp_) FUNCTION DNRM2(n, x, incx)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: incx, n
          REAL(rp_) :: x(*)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          REAL(rp_) :: absxi, norm, scale, ssq
          INTEGER(ip_) :: ix
          INTRINSIC :: ABS, SQRT
          IF (n<1 .OR. incx<1) THEN
            norm = zero
          ELSE IF (n==1) THEN
            norm = ABS(x(1))
          ELSE
            scale = zero
            ssq = one
            DO ix = 1, 1 + (n-1)*incx, incx
              IF (x(ix)/=zero) THEN
                absxi = ABS(x(ix))
                IF (scale<absxi) THEN
                  ssq = one + ssq*(scale/absxi)**2
                  scale = absxi
                ELSE
                  ssq = ssq + (absxi/scale)**2
                END IF
              END IF
            END DO
            norm = scale*SQRT(ssq)
          END IF
          DNRM2 = norm
          RETURN
        END FUNCTION

        SUBROUTINE DROT(n, dx, incx, dy, incy, c, s)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: c, s
          INTEGER(ip_) :: incx, incy, n
          REAL(rp_) :: dx(*), dy(*)
          REAL(rp_) :: dtemp
          INTEGER(ip_) :: i, ix, iy
          IF (n<=0) RETURN
          IF (incx==1 .AND. incy==1) THEN
            DO i = 1, n
              dtemp = c*dx(i) + s*dy(i)
              dy(i) = c*dy(i) - s*dx(i)
              dx(i) = dtemp
            END DO
          ELSE
            ix = 1
            iy = 1
            IF (incx<0) ix = (-n+1)*incx + 1
            IF (incy<0) iy = (-n+1)*incy + 1
            DO i = 1, n
              dtemp = c*dx(ix) + s*dy(iy)
              dy(iy) = c*dy(iy) - s*dx(ix)
              dx(ix) = dtemp
              ix = ix + incx
              iy = iy + incy
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DROTG(da, db, c, s)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: c, da, db, s
          REAL(rp_) :: r, roe, scale, z
          INTRINSIC :: ABS, SIGN, SQRT
          scale = ABS(da) + ABS(db)
          IF (scale==0.0_rp_) THEN
            c = 1.0_rp_
            s = 0.0_rp_
            r = 0.0_rp_
            z = 0.0_rp_
          ELSE
            roe = db
            IF (ABS(da)>ABS(db)) roe = da
            r = scale*SQRT((da/scale)**2+(db/scale)**2)
            r = SIGN(1.0_rp_, roe)*r
            c = da/r
            s = db/r
            z = 1.0_rp_
            IF (ABS(da)>ABS(db)) z = s
            IF (ABS(db)>=ABS(da) .AND. c/=0.0_rp_) z = 1.0_rp_/c
          END IF
          da = r
          db = z
          RETURN
        END SUBROUTINE

        SUBROUTINE DSCAL(n, da, dx, incx)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: da
          INTEGER(ip_) :: incx, n
          REAL(rp_) :: dx(*)
          INTEGER(ip_) :: i, m, mp1, nincx
          INTRINSIC :: MOD
          IF (n<=0 .OR. incx<=0) RETURN
          IF (incx==1) THEN
            m = MOD(n, 5)
            IF (m/=0) THEN
              DO i = 1, m
                dx(i) = da*dx(i)
              END DO
              IF (n<5) RETURN
            END IF
            mp1 = m + 1
            DO i = mp1, n, 5
              dx(i) = da*dx(i)
              dx(i+1) = da*dx(i+1)
              dx(i+2) = da*dx(i+2)
              dx(i+3) = da*dx(i+3)
              dx(i+4) = da*dx(i+4)
            END DO
          ELSE
            nincx = n*incx
            DO i = 1, nincx, incx
              dx(i) = da*dx(i)
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DSPMV(uplo, n, alpha, ap, x, incx, beta, y, incy)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: alpha, beta
          INTEGER(ip_) :: incx, incy, n
          CHARACTER :: uplo
          REAL(rp_) :: ap(*), x(*), y(*)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          REAL(rp_) :: temp1, temp2
          INTEGER(ip_) :: i, info, ix, iy, j, jx, jy, k, kk, kx, ky
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          info = 0
          IF (.NOT. LSAME(uplo,'U') .AND. .NOT. LSAME(uplo,'L')) THEN
            info = 1
          ELSE IF (n<0) THEN
            info = 2
          ELSE IF (incx==0) THEN
            info = 6
          ELSE IF (incy==0) THEN
            info = 9
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('SPMV', info)
            RETURN
          END IF
          IF ((n==0) .OR. ((alpha==zero) .AND. (beta==one))) RETURN
          IF (incx>0) THEN
            kx = 1
          ELSE
            kx = 1 - (n-1)*incx
          END IF
          IF (incy>0) THEN
            ky = 1
          ELSE
            ky = 1 - (n-1)*incy
          END IF
          IF (beta/=one) THEN
            IF (incy==1) THEN
              IF (beta==zero) THEN
                DO i = 1, n
                  y(i) = zero
                END DO
              ELSE
                DO i = 1, n
                  y(i) = beta*y(i)
                END DO
              END IF
            ELSE
              iy = ky
              IF (beta==zero) THEN
                DO i = 1, n
                  y(iy) = zero
                  iy = iy + incy
                END DO
              ELSE
                DO i = 1, n
                  y(iy) = beta*y(iy)
                  iy = iy + incy
                END DO
              END IF
            END IF
          END IF
          IF (alpha==zero) RETURN
          kk = 1
          IF (LSAME(uplo,'U')) THEN
            IF ((incx==1) .AND. (incy==1)) THEN
              DO j = 1, n
                temp1 = alpha*x(j)
                temp2 = zero
                k = kk
                DO i = 1, j - 1
                  y(i) = y(i) + temp1*ap(k)
                  temp2 = temp2 + ap(k)*x(i)
                  k = k + 1
                END DO
                y(j) = y(j) + temp1*ap(kk+j-1) + alpha*temp2
                kk = kk + j
              END DO
            ELSE
              jx = kx
              jy = ky
              DO j = 1, n
                temp1 = alpha*x(jx)
                temp2 = zero
                ix = kx
                iy = ky
                DO k = kk, kk + j - 2
                  y(iy) = y(iy) + temp1*ap(k)
                  temp2 = temp2 + ap(k)*x(ix)
                  ix = ix + incx
                  iy = iy + incy
                END DO
                y(jy) = y(jy) + temp1*ap(kk+j-1) + alpha*temp2
                jx = jx + incx
                jy = jy + incy
                kk = kk + j
              END DO
            END IF
          ELSE
            IF ((incx==1) .AND. (incy==1)) THEN
              DO j = 1, n
                temp1 = alpha*x(j)
                temp2 = zero
                y(j) = y(j) + temp1*ap(kk)
                k = kk + 1
                DO i = j + 1, n
                  y(i) = y(i) + temp1*ap(k)
                  temp2 = temp2 + ap(k)*x(i)
                  k = k + 1
                END DO
                y(j) = y(j) + alpha*temp2
                kk = kk + (n-j+1)
              END DO
            ELSE
              jx = kx
              jy = ky
              DO j = 1, n
                temp1 = alpha*x(jx)
                temp2 = zero
                y(jy) = y(jy) + temp1*ap(kk)
                ix = jx
                iy = jy
                DO k = kk + 1, kk + n - j
                  ix = ix + incx
                  iy = iy + incy
                  y(iy) = y(iy) + temp1*ap(k)
                  temp2 = temp2 + ap(k)*x(ix)
                END DO
                y(jy) = y(jy) + alpha*temp2
                jx = jx + incx
                jy = jy + incy
                kk = kk + (n-j+1)
              END DO
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DSWAP(n, dx, incx, dy, incy)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: incx, incy, n
          REAL(rp_) :: dx(*), dy(*)
          REAL(rp_) :: dtemp
          INTEGER(ip_) :: i, ix, iy, m, mp1
          INTRINSIC :: MOD
          IF (n<=0) RETURN
          IF (incx==1 .AND. incy==1) THEN
            m = MOD(n, 3_ip_)
            IF (m/=0) THEN
              DO i = 1, m
                dtemp = dx(i)
                dx(i) = dy(i)
                dy(i) = dtemp
              END DO
              IF (n<3) RETURN
            END IF
            mp1 = m + 1
            DO i = mp1, n, 3
              dtemp = dx(i)
              dx(i) = dy(i)
              dy(i) = dtemp
              dtemp = dx(i+1)
              dx(i+1) = dy(i+1)
              dy(i+1) = dtemp
              dtemp = dx(i+2)
              dx(i+2) = dy(i+2)
              dy(i+2) = dtemp
            END DO
          ELSE
            ix = 1
            iy = 1
            IF (incx<0) ix = (-n+1)*incx + 1
            IF (incy<0) iy = (-n+1)*incy + 1
            DO i = 1, n
              dtemp = dx(ix)
              dx(ix) = dy(iy)
              dy(iy) = dtemp
              ix = ix + incx
              iy = iy + incy
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DSYMM(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c,  &
          ldc)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: alpha, beta
          INTEGER(ip_) :: lda, ldb, ldc, m, n
          CHARACTER :: side, uplo
          REAL(rp_) :: a(lda, *), b(ldb, *), c(ldc, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          INTRINSIC :: MAX
          REAL(rp_) :: temp1, temp2
          INTEGER(ip_) :: i, info, j, k, nrowa
          LOGICAL :: upper
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          IF (LSAME(side,'L')) THEN
            nrowa = m
          ELSE
            nrowa = n
          END IF
          upper = LSAME(uplo, 'U')
          info = 0
          IF ((.NOT. LSAME(side,'L')) .AND. (.NOT. LSAME(side,'R'))) &
            THEN
            info = 1
          ELSE IF ((.NOT. upper) .AND. (.NOT. LSAME(uplo,'L'))) THEN
            info = 2
          ELSE IF (m<0) THEN
            info = 3
          ELSE IF (n<0) THEN
            info = 4
          ELSE IF (lda<MAX(1,nrowa)) THEN
            info = 7
          ELSE IF (ldb<MAX(1,m)) THEN
            info = 9
          ELSE IF (ldc<MAX(1,m)) THEN
            info = 12
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('SYMM', info)
            RETURN
          END IF
          IF ((m==0) .OR. (n==0) .OR. ((alpha==zero) .AND. (beta== one)))   &
            RETURN
          IF (alpha==zero) THEN
            IF (beta==zero) THEN
              DO j = 1, n
                DO i = 1, m
                  c(i, j) = zero
                END DO
              END DO
            ELSE
              DO j = 1, n
                DO i = 1, m
                  c(i, j) = beta*c(i, j)
                END DO
              END DO
            END IF
            RETURN
          END IF
          IF (LSAME(side,'L')) THEN
            IF (upper) THEN
              DO j = 1, n
                DO i = 1, m
                  temp1 = alpha*b(i, j)
                  temp2 = zero
                  DO k = 1, i - 1
                    c(k, j) = c(k, j) + temp1*a(k, i)
                    temp2 = temp2 + b(k, j)*a(k, i)
                  END DO
                  IF (beta==zero) THEN
                    c(i, j) = temp1*a(i, i) + alpha*temp2
                  ELSE
                    c(i, j) = beta*c(i, j) + temp1*a(i, i) + alpha*temp2
                  END IF
                END DO
              END DO
            ELSE
              DO j = 1, n
                DO i = m, 1_ip_, -1_ip_
                  temp1 = alpha*b(i, j)
                  temp2 = zero
                  DO k = i + 1, m
                    c(k, j) = c(k, j) + temp1*a(k, i)
                    temp2 = temp2 + b(k, j)*a(k, i)
                  END DO
                  IF (beta==zero) THEN
                    c(i, j) = temp1*a(i, i) + alpha*temp2
                  ELSE
                    c(i, j) = beta*c(i, j) + temp1*a(i, i) + alpha*temp2
                  END IF
                END DO
              END DO
            END IF
          ELSE
            DO j = 1, n
              temp1 = alpha*a(j, j)
              IF (beta==zero) THEN
                DO i = 1, m
                  c(i, j) = temp1*b(i, j)
                END DO
              ELSE
                DO i = 1, m
                  c(i, j) = beta*c(i, j) + temp1*b(i, j)
                END DO
              END IF
              DO k = 1, j - 1
                IF (upper) THEN
                  temp1 = alpha*a(k, j)
                ELSE
                  temp1 = alpha*a(j, k)
                END IF
                DO i = 1, m
                  c(i, j) = c(i, j) + temp1*b(i, k)
                END DO
              END DO
              DO k = j + 1, n
                IF (upper) THEN
                  temp1 = alpha*a(j, k)
                ELSE
                  temp1 = alpha*a(k, j)
                END IF
                DO i = 1, m
                  c(i, j) = c(i, j) + temp1*b(i, k)
                END DO
              END DO
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DSYMV(uplo, n, alpha, a, lda, x, incx, beta, y, incy)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: alpha, beta
          INTEGER(ip_) :: incx, incy, lda, n
          CHARACTER :: uplo
          REAL(rp_) :: a(lda, *), x(*), y(*)
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          REAL(rp_) :: temp1, temp2
          INTEGER(ip_) :: i, info, ix, iy, j, jx, jy, kx, ky
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          INTRINSIC :: MAX
          info = 0
          IF (.NOT. LSAME(uplo,'U') .AND. .NOT. LSAME(uplo,'L')) THEN
            info = 1
          ELSE IF (n<0) THEN
            info = 2
          ELSE IF (lda<MAX(1,n)) THEN
            info = 5
          ELSE IF (incx==0) THEN
            info = 7
          ELSE IF (incy==0) THEN
            info = 10
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('SYMV', info)
            RETURN
          END IF
          IF ((n==0) .OR. ((alpha==zero) .AND. (beta==one))) RETURN
          IF (incx>0) THEN
            kx = 1
          ELSE
            kx = 1 - (n-1)*incx
          END IF
          IF (incy>0) THEN
            ky = 1
          ELSE
            ky = 1 - (n-1)*incy
          END IF
          IF (beta/=one) THEN
            IF (incy==1) THEN
              IF (beta==zero) THEN
                DO i = 1, n
                  y(i) = zero
                END DO
              ELSE
                DO i = 1, n
                  y(i) = beta*y(i)
                END DO
              END IF
            ELSE
              iy = ky
              IF (beta==zero) THEN
                DO i = 1, n
                  y(iy) = zero
                  iy = iy + incy
                END DO
              ELSE
                DO i = 1, n
                  y(iy) = beta*y(iy)
                  iy = iy + incy
                END DO
              END IF
            END IF
          END IF
          IF (alpha==zero) RETURN
          IF (LSAME(uplo,'U')) THEN
            IF ((incx==1) .AND. (incy==1)) THEN
              DO j = 1, n
                temp1 = alpha*x(j)
                temp2 = zero
                DO i = 1, j - 1
                  y(i) = y(i) + temp1*a(i, j)
                  temp2 = temp2 + a(i, j)*x(i)
                END DO
                y(j) = y(j) + temp1*a(j, j) + alpha*temp2
              END DO
            ELSE
              jx = kx
              jy = ky
              DO j = 1, n
                temp1 = alpha*x(jx)
                temp2 = zero
                ix = kx
                iy = ky
                DO i = 1, j - 1
                  y(iy) = y(iy) + temp1*a(i, j)
                  temp2 = temp2 + a(i, j)*x(ix)
                  ix = ix + incx
                  iy = iy + incy
                END DO
                y(jy) = y(jy) + temp1*a(j, j) + alpha*temp2
                jx = jx + incx
                jy = jy + incy
              END DO
            END IF
          ELSE
            IF ((incx==1) .AND. (incy==1)) THEN
              DO j = 1, n
                temp1 = alpha*x(j)
                temp2 = zero
                y(j) = y(j) + temp1*a(j, j)
                DO i = j + 1, n
                  y(i) = y(i) + temp1*a(i, j)
                  temp2 = temp2 + a(i, j)*x(i)
                END DO
                y(j) = y(j) + alpha*temp2
              END DO
            ELSE
              jx = kx
              jy = ky
              DO j = 1, n
                temp1 = alpha*x(jx)
                temp2 = zero
                y(jy) = y(jy) + temp1*a(j, j)
                ix = jx
                iy = jy
                DO i = j + 1, n
                  ix = ix + incx
                  iy = iy + incy
                  y(iy) = y(iy) + temp1*a(i, j)
                  temp2 = temp2 + a(i, j)*x(ix)
                END DO
                y(jy) = y(jy) + alpha*temp2
                jx = jx + incx
                jy = jy + incy
              END DO
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DSYR(uplo, n, alpha, x, incx, a, lda)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: alpha
          INTEGER(ip_) :: incx, lda, n
          CHARACTER :: uplo
          REAL(rp_) :: a(lda, *), x(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          REAL(rp_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, kx
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          INTRINSIC :: MAX
          info = 0
          IF (.NOT. LSAME(uplo,'U') .AND. .NOT. LSAME(uplo,'L')) THEN
            info = 1
          ELSE IF (n<0) THEN
            info = 2
          ELSE IF (incx==0) THEN
            info = 5
          ELSE IF (lda<MAX(1,n)) THEN
            info = 7
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('SYR', info)
            RETURN
          END IF
          IF ((n==0) .OR. (alpha==zero)) RETURN
          IF (incx<=0) THEN
            kx = 1 - (n-1)*incx
          ELSE IF (incx/=1) THEN
            kx = 1
          END IF
          IF (LSAME(uplo,'U')) THEN
            IF (incx==1) THEN
              DO j = 1, n
                IF (x(j)/=zero) THEN
                  temp = alpha*x(j)
                  DO i = 1, j
                    a(i, j) = a(i, j) + x(i)*temp
                  END DO
                END IF
              END DO
            ELSE
              jx = kx
              DO j = 1, n
                IF (x(jx)/=zero) THEN
                  temp = alpha*x(jx)
                  ix = kx
                  DO i = 1, j
                    a(i, j) = a(i, j) + x(ix)*temp
                    ix = ix + incx
                  END DO
                END IF
                jx = jx + incx
              END DO
            END IF
          ELSE
            IF (incx==1) THEN
              DO j = 1, n
                IF (x(j)/=zero) THEN
                  temp = alpha*x(j)
                  DO i = j, n
                    a(i, j) = a(i, j) + x(i)*temp
                  END DO
                END IF
              END DO
            ELSE
              jx = kx
              DO j = 1, n
                IF (x(jx)/=zero) THEN
                  temp = alpha*x(jx)
                  ix = jx
                  DO i = j, n
                    a(i, j) = a(i, j) + x(ix)*temp
                    ix = ix + incx
                  END DO
                END IF
                jx = jx + incx
              END DO
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DSYR2(uplo, n, alpha, x, incx, y, incy, a, lda)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: alpha
          INTEGER(ip_) :: incx, incy, lda, n
          CHARACTER :: uplo
          REAL(rp_) :: a(lda, *), x(*), y(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          REAL(rp_) :: temp1, temp2
          INTEGER(ip_) :: i, info, ix, iy, j, jx, jy, kx, ky
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          INTRINSIC :: MAX
          info = 0
          IF (.NOT. LSAME(uplo,'U') .AND. .NOT. LSAME(uplo,'L')) THEN
            info = 1
          ELSE IF (n<0) THEN
            info = 2
          ELSE IF (incx==0) THEN
            info = 5
          ELSE IF (incy==0) THEN
            info = 7
          ELSE IF (lda<MAX(1,n)) THEN
            info = 9
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('SYR2', info)
            RETURN
          END IF
          IF ((n==0) .OR. (alpha==zero)) RETURN
          IF ((incx/=1) .OR. (incy/=1)) THEN
            IF (incx>0) THEN
              kx = 1
            ELSE
              kx = 1 - (n-1)*incx
            END IF
            IF (incy>0) THEN
              ky = 1
            ELSE
              ky = 1 - (n-1)*incy
            END IF
            jx = kx
            jy = ky
          END IF
          IF (LSAME(uplo,'U')) THEN
            IF ((incx==1) .AND. (incy==1)) THEN
              DO j = 1, n
                IF ((x(j)/=zero) .OR. (y(j)/=zero)) THEN
                  temp1 = alpha*y(j)
                  temp2 = alpha*x(j)
                  DO i = 1, j
                    a(i, j) = a(i, j) + x(i)*temp1 + y(i)*temp2
                  END DO
                END IF
              END DO
            ELSE
              DO j = 1, n
                IF ((x(jx)/=zero) .OR. (y(jy)/=zero)) THEN
                  temp1 = alpha*y(jy)
                  temp2 = alpha*x(jx)
                  ix = kx
                  iy = ky
                  DO i = 1, j
                    a(i, j) = a(i, j) + x(ix)*temp1 + y(iy)*temp2
                    ix = ix + incx
                    iy = iy + incy
                  END DO
                END IF
                jx = jx + incx
                jy = jy + incy
              END DO
            END IF
          ELSE
            IF ((incx==1) .AND. (incy==1)) THEN
              DO j = 1, n
                IF ((x(j)/=zero) .OR. (y(j)/=zero)) THEN
                  temp1 = alpha*y(j)
                  temp2 = alpha*x(j)
                  DO i = j, n
                    a(i, j) = a(i, j) + x(i)*temp1 + y(i)*temp2
                  END DO
                END IF
              END DO
            ELSE
              DO j = 1, n
                IF ((x(jx)/=zero) .OR. (y(jy)/=zero)) THEN
                  temp1 = alpha*y(jy)
                  temp2 = alpha*x(jx)
                  ix = jx
                  iy = jy
                  DO i = j, n
                    a(i, j) = a(i, j) + x(ix)*temp1 + y(iy)*temp2
                    ix = ix + incx
                    iy = iy + incy
                  END DO
                END IF
                jx = jx + incx
                jy = jy + incy
              END DO
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DSYR2K(uplo, trans, n, k, alpha, a, lda, b, ldb, beta,   &
          c, ldc)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: alpha, beta
          INTEGER(ip_) :: k, lda, ldb, ldc, n
          CHARACTER :: trans, uplo
          REAL(rp_) :: a(lda, *), b(ldb, *), c(ldc, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          INTRINSIC :: MAX
          REAL(rp_) :: temp1, temp2
          INTEGER(ip_) :: i, info, j, l, nrowa
          LOGICAL :: upper
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          IF (LSAME(trans,'N')) THEN
            nrowa = n
          ELSE
            nrowa = k
          END IF
          upper = LSAME(uplo, 'U')
          info = 0
          IF ((.NOT. upper) .AND. (.NOT. LSAME(uplo,'L'))) THEN
            info = 1
          ELSE IF ((.NOT. LSAME(trans,'N')) .AND. (.NOT. &
            LSAME(trans, 'T')) .AND. (.NOT. LSAME(trans,'C'))) THEN
            info = 2
          ELSE IF (n<0) THEN
            info = 3
          ELSE IF (k<0) THEN
            info = 4
          ELSE IF (lda<MAX(1,nrowa)) THEN
            info = 7
          ELSE IF (ldb<MAX(1,nrowa)) THEN
            info = 9
          ELSE IF (ldc<MAX(1,n)) THEN
            info = 12
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('SYR2K', info)
            RETURN
          END IF
          IF ((n==0) .OR. (((alpha==zero) .OR. (k==0)) .AND. (beta==        &
            one))) RETURN
          IF (alpha==zero) THEN
            IF (upper) THEN
              IF (beta==zero) THEN
                DO j = 1, n
                  DO i = 1, j
                    c(i, j) = zero
                  END DO
                END DO
              ELSE
                DO j = 1, n
                  DO i = 1, j
                    c(i, j) = beta*c(i, j)
                  END DO
                END DO
              END IF
            ELSE
              IF (beta==zero) THEN
                DO j = 1, n
                  DO i = j, n
                    c(i, j) = zero
                  END DO
                END DO
              ELSE
                DO j = 1, n
                  DO i = j, n
                    c(i, j) = beta*c(i, j)
                  END DO
                END DO
              END IF
            END IF
            RETURN
          END IF
          IF (LSAME(trans,'N')) THEN
            IF (upper) THEN
              DO j = 1, n
                IF (beta==zero) THEN
                  DO i = 1, j
                    c(i, j) = zero
                  END DO
                ELSE IF (beta/=one) THEN
                  DO i = 1, j
                    c(i, j) = beta*c(i, j)
                  END DO
                END IF
                DO l = 1, k
                  IF ((a(j,l)/=zero) .OR. (b(j,l)/=zero)) THEN
                    temp1 = alpha*b(j, l)
                    temp2 = alpha*a(j, l)
                    DO i = 1, j
                      c(i, j) = c(i, j) + a(i, l)*temp1 + b(i, l)*temp2
                    END DO
                  END IF
                END DO
              END DO
            ELSE
              DO j = 1, n
                IF (beta==zero) THEN
                  DO i = j, n
                    c(i, j) = zero
                  END DO
                ELSE IF (beta/=one) THEN
                  DO i = j, n
                    c(i, j) = beta*c(i, j)
                  END DO
                END IF
                DO l = 1, k
                  IF ((a(j,l)/=zero) .OR. (b(j,l)/=zero)) THEN
                    temp1 = alpha*b(j, l)
                    temp2 = alpha*a(j, l)
                    DO i = j, n
                      c(i, j) = c(i, j) + a(i, l)*temp1 + b(i, l)*temp2
                    END DO
                  END IF
                END DO
              END DO
            END IF
          ELSE
            IF (upper) THEN
              DO j = 1, n
                DO i = 1, j
                  temp1 = zero
                  temp2 = zero
                  DO l = 1, k
                    temp1 = temp1 + a(l, i)*b(l, j)
                    temp2 = temp2 + b(l, i)*a(l, j)
                  END DO
                  IF (beta==zero) THEN
                    c(i, j) = alpha*temp1 + alpha*temp2
                  ELSE
                    c(i, j) = beta*c(i, j) + alpha*temp1 + alpha*temp2
                  END IF
                END DO
              END DO
            ELSE
              DO j = 1, n
                DO i = j, n
                  temp1 = zero
                  temp2 = zero
                  DO l = 1, k
                    temp1 = temp1 + a(l, i)*b(l, j)
                    temp2 = temp2 + b(l, i)*a(l, j)
                  END DO
                  IF (beta==zero) THEN
                    c(i, j) = alpha*temp1 + alpha*temp2
                  ELSE
                    c(i, j) = beta*c(i, j) + alpha*temp1 + alpha*temp2
                  END IF
                END DO
              END DO
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DSYRK(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: alpha, beta
          INTEGER(ip_) :: k, lda, ldc, n
          CHARACTER :: trans, uplo
          REAL(rp_) :: a(lda, *), c(ldc, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          INTRINSIC :: MAX
          REAL(rp_) :: temp
          INTEGER(ip_) :: i, info, j, l, nrowa
          LOGICAL :: upper
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          IF (LSAME(trans,'N')) THEN
            nrowa = n
          ELSE
            nrowa = k
          END IF
          upper = LSAME(uplo, 'U')
          info = 0
          IF ((.NOT. upper) .AND. (.NOT. LSAME(uplo,'L'))) THEN
            info = 1
          ELSE IF ((.NOT. LSAME(trans,'N')) .AND. (.NOT. &
            LSAME(trans, 'T')) .AND. (.NOT. LSAME(trans,'C'))) THEN
            info = 2
          ELSE IF (n<0) THEN
            info = 3
          ELSE IF (k<0) THEN
            info = 4
          ELSE IF (lda<MAX(1,nrowa)) THEN
            info = 7
          ELSE IF (ldc<MAX(1,n)) THEN
            info = 10
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('SYRK', info)
            RETURN
          END IF
          IF ((n==0) .OR. (((alpha==zero) .OR. (k==0)) .AND. (beta==        &
            one))) RETURN
          IF (alpha==zero) THEN
            IF (upper) THEN
              IF (beta==zero) THEN
                DO j = 1, n
                  DO i = 1, j
                    c(i, j) = zero
                  END DO
                END DO
              ELSE
                DO j = 1, n
                  DO i = 1, j
                    c(i, j) = beta*c(i, j)
                  END DO
                END DO
              END IF
            ELSE
              IF (beta==zero) THEN
                DO j = 1, n
                  DO i = j, n
                    c(i, j) = zero
                  END DO
                END DO
              ELSE
                DO j = 1, n
                  DO i = j, n
                    c(i, j) = beta*c(i, j)
                  END DO
                END DO
              END IF
            END IF
            RETURN
          END IF
          IF (LSAME(trans,'N')) THEN
            IF (upper) THEN
              DO j = 1, n
                IF (beta==zero) THEN
                  DO i = 1, j
                    c(i, j) = zero
                  END DO
                ELSE IF (beta/=one) THEN
                  DO i = 1, j
                    c(i, j) = beta*c(i, j)
                  END DO
                END IF
                DO l = 1, k
                  IF (a(j,l)/=zero) THEN
                    temp = alpha*a(j, l)
                    DO i = 1, j
                      c(i, j) = c(i, j) + temp*a(i, l)
                    END DO
                  END IF
                END DO
              END DO
            ELSE
              DO j = 1, n
                IF (beta==zero) THEN
                  DO i = j, n
                    c(i, j) = zero
                  END DO
                ELSE IF (beta/=one) THEN
                  DO i = j, n
                    c(i, j) = beta*c(i, j)
                  END DO
                END IF
                DO l = 1, k
                  IF (a(j,l)/=zero) THEN
                    temp = alpha*a(j, l)
                    DO i = j, n
                      c(i, j) = c(i, j) + temp*a(i, l)
                    END DO
                  END IF
                END DO
              END DO
            END IF
          ELSE
            IF (upper) THEN
              DO j = 1, n
                DO i = 1, j
                  temp = zero
                  DO l = 1, k
                    temp = temp + a(l, i)*a(l, j)
                  END DO
                  IF (beta==zero) THEN
                    c(i, j) = alpha*temp
                  ELSE
                    c(i, j) = alpha*temp + beta*c(i, j)
                  END IF
                END DO
              END DO
            ELSE
              DO j = 1, n
                DO i = j, n
                  temp = zero
                  DO l = 1, k
                    temp = temp + a(l, i)*a(l, j)
                  END DO
                  IF (beta==zero) THEN
                    c(i, j) = alpha*temp
                  ELSE
                    c(i, j) = alpha*temp + beta*c(i, j)
                  END IF
                END DO
              END DO
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DTBSV(uplo, trans, diag, n, k, a, lda, x, incx)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: incx, k, lda, n
          CHARACTER :: diag, trans, uplo
          REAL(rp_) :: a(lda, *), x(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          REAL(rp_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, kplus1, kx, l
          LOGICAL :: nounit
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          INTRINSIC :: MAX, MIN
          info = 0
          IF (.NOT. LSAME(uplo,'U') .AND. .NOT. LSAME(uplo,'L')) THEN
            info = 1
          ELSE IF (.NOT. LSAME(trans,'N') .AND. .NOT. LSAME(trans,& 
            'T') .AND. .NOT. LSAME(trans,'C')) THEN
            info = 2
          ELSE IF (.NOT. LSAME(diag,'U') .AND. .NOT. LSAME(diag,'N')) &
            THEN
            info = 3
          ELSE IF (n<0) THEN
            info = 4
          ELSE IF (k<0) THEN
            info = 5
          ELSE IF (lda<(k+1)) THEN
            info = 7
          ELSE IF (incx==0) THEN
            info = 9
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('TBSV', info)
            RETURN
          END IF
          IF (n==0) RETURN
          nounit = LSAME(diag, 'N')
          IF (incx<=0) THEN
            kx = 1 - (n-1)*incx
          ELSE IF (incx/=1) THEN
            kx = 1
          END IF
          IF (LSAME(trans,'N')) THEN
            IF (LSAME(uplo,'U')) THEN
              kplus1 = k + 1
              IF (incx==1) THEN
                DO j = n, 1_ip_, -1_ip_
                  IF (x(j)/=zero) THEN
                    l = kplus1 - j
                    IF (nounit) x(j) = x(j)/a(kplus1, j)
                    temp = x(j)
                    DO i = j - 1, MAX(1, j-k), -1_ip_
                      x(i) = x(i) - temp*a(l+i, j)
                    END DO
                  END IF
                END DO
              ELSE
                kx = kx + (n-1)*incx
                jx = kx
                DO j = n, 1_ip_, -1_ip_
                  kx = kx - incx
                  IF (x(jx)/=zero) THEN
                    ix = kx
                    l = kplus1 - j
                    IF (nounit) x(jx) = x(jx)/a(kplus1, j)
                    temp = x(jx)
                    DO i = j - 1, MAX(1, j-k), -1_ip_
                      x(ix) = x(ix) - temp*a(l+i, j)
                      ix = ix - incx
                    END DO
                  END IF
                  jx = jx - incx
                END DO
              END IF
            ELSE
              IF (incx==1) THEN
                DO j = 1, n
                  IF (x(j)/=zero) THEN
                    l = 1 - j
                    IF (nounit) x(j) = x(j)/a(1, j)
                    temp = x(j)
                    DO i = j + 1, MIN(n, j+k)
                      x(i) = x(i) - temp*a(l+i, j)
                    END DO
                  END IF
                END DO
              ELSE
                jx = kx
                DO j = 1, n
                  kx = kx + incx
                  IF (x(jx)/=zero) THEN
                    ix = kx
                    l = 1 - j
                    IF (nounit) x(jx) = x(jx)/a(1, j)
                    temp = x(jx)
                    DO i = j + 1, MIN(n, j+k)
                      x(ix) = x(ix) - temp*a(l+i, j)
                      ix = ix + incx
                    END DO
                  END IF
                  jx = jx + incx
                END DO
              END IF
            END IF
          ELSE
            IF (LSAME(uplo,'U')) THEN
              kplus1 = k + 1
              IF (incx==1) THEN
                DO j = 1, n
                  temp = x(j)
                  l = kplus1 - j
                  DO i = MAX(1, j-k), j - 1
                    temp = temp - a(l+i, j)*x(i)
                  END DO
                  IF (nounit) temp = temp/a(kplus1, j)
                  x(j) = temp
                END DO
              ELSE
                jx = kx
                DO j = 1, n
                  temp = x(jx)
                  ix = kx
                  l = kplus1 - j
                  DO i = MAX(1, j-k), j - 1
                    temp = temp - a(l+i, j)*x(ix)
                    ix = ix + incx
                  END DO
                  IF (nounit) temp = temp/a(kplus1, j)
                  x(jx) = temp
                  jx = jx + incx
                  IF (j>k) kx = kx + incx
                END DO
              END IF
            ELSE
              IF (incx==1) THEN
                DO j = n, 1_ip_, -1_ip_
                  temp = x(j)
                  l = 1 - j
                  DO i = MIN(n, j+k), j + 1, -1_ip_
                    temp = temp - a(l+i, j)*x(i)
                  END DO
                  IF (nounit) temp = temp/a(1, j)
                  x(j) = temp
                END DO
              ELSE
                kx = kx + (n-1)*incx
                jx = kx
                DO j = n, 1_ip_, -1_ip_
                  temp = x(jx)
                  ix = kx
                  l = 1 - j
                  DO i = MIN(n, j+k), j + 1, -1_ip_
                    temp = temp - a(l+i, j)*x(ix)
                    ix = ix - incx
                  END DO
                  IF (nounit) temp = temp/a(1, j)
                  x(jx) = temp
                  jx = jx - incx
                  IF ((n-j)>=k) kx = kx - incx
                END DO
              END IF
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DTPMV(uplo, trans, diag, n, ap, x, incx)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: incx, n
          CHARACTER :: diag, trans, uplo
          REAL(rp_) :: ap(*), x(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          REAL(rp_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, k, kk, kx
          LOGICAL :: nounit
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          info = 0
          IF (.NOT. LSAME(uplo,'U') .AND. .NOT. LSAME(uplo,'L')) THEN
            info = 1
          ELSE IF (.NOT. LSAME(trans,'N') .AND. .NOT. LSAME(trans,& 
            'T') .AND. .NOT. LSAME(trans,'C')) THEN
            info = 2
          ELSE IF (.NOT. LSAME(diag,'U') .AND. .NOT. LSAME(diag,'N')) &
            THEN
            info = 3
          ELSE IF (n<0) THEN
            info = 4
          ELSE IF (incx==0) THEN
            info = 7
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('TPMV', info)
            RETURN
          END IF
          IF (n==0) RETURN
          nounit = LSAME(diag, 'N')
          IF (incx<=0) THEN
            kx = 1 - (n-1)*incx
          ELSE IF (incx/=1) THEN
            kx = 1
          END IF
          IF (LSAME(trans,'N')) THEN
            IF (LSAME(uplo,'U')) THEN
              kk = 1
              IF (incx==1) THEN
                DO j = 1, n
                  IF (x(j)/=zero) THEN
                    temp = x(j)
                    k = kk
                    DO i = 1, j - 1
                      x(i) = x(i) + temp*ap(k)
                      k = k + 1
                    END DO
                    IF (nounit) x(j) = x(j)*ap(kk+j-1)
                  END IF
                  kk = kk + j
                END DO
              ELSE
                jx = kx
                DO j = 1, n
                  IF (x(jx)/=zero) THEN
                    temp = x(jx)
                    ix = kx
                    DO k = kk, kk + j - 2
                      x(ix) = x(ix) + temp*ap(k)
                      ix = ix + incx
                    END DO
                    IF (nounit) x(jx) = x(jx)*ap(kk+j-1)
                  END IF
                  jx = jx + incx
                  kk = kk + j
                END DO
              END IF
            ELSE
              kk = (n*(n+1))/2
              IF (incx==1) THEN
                DO j = n, 1_ip_, -1_ip_
                  IF (x(j)/=zero) THEN
                    temp = x(j)
                    k = kk
                    DO i = n, j + 1, -1_ip_
                      x(i) = x(i) + temp*ap(k)
                      k = k - 1
                    END DO
                    IF (nounit) x(j) = x(j)*ap(kk-n+j)
                  END IF
                  kk = kk - (n-j+1)
                END DO
              ELSE
                kx = kx + (n-1)*incx
                jx = kx
                DO j = n, 1_ip_, -1_ip_
                  IF (x(jx)/=zero) THEN
                    temp = x(jx)
                    ix = kx
                    DO k = kk, kk - (n-(j+1)), -1_ip_
                      x(ix) = x(ix) + temp*ap(k)
                      ix = ix - incx
                    END DO
                    IF (nounit) x(jx) = x(jx)*ap(kk-n+j)
                  END IF
                  jx = jx - incx
                  kk = kk - (n-j+1)
                END DO
              END IF
            END IF
          ELSE
            IF (LSAME(uplo,'U')) THEN
              kk = (n*(n+1))/2
              IF (incx==1) THEN
                DO j = n, 1_ip_, -1_ip_
                  temp = x(j)
                  IF (nounit) temp = temp*ap(kk)
                  k = kk - 1
                  DO i = j - 1, 1_ip_, -1_ip_
                    temp = temp + ap(k)*x(i)
                    k = k - 1
                  END DO
                  x(j) = temp
                  kk = kk - j
                END DO
              ELSE
                jx = kx + (n-1)*incx
                DO j = n, 1_ip_, -1_ip_
                  temp = x(jx)
                  ix = jx
                  IF (nounit) temp = temp*ap(kk)
                  DO k = kk - 1, kk - j + 1, -1_ip_
                    ix = ix - incx
                    temp = temp + ap(k)*x(ix)
                  END DO
                  x(jx) = temp
                  jx = jx - incx
                  kk = kk - j
                END DO
              END IF
            ELSE
              kk = 1
              IF (incx==1) THEN
                DO j = 1, n
                  temp = x(j)
                  IF (nounit) temp = temp*ap(kk)
                  k = kk + 1
                  DO i = j + 1, n
                    temp = temp + ap(k)*x(i)
                    k = k + 1
                  END DO
                  x(j) = temp
                  kk = kk + (n-j+1)
                END DO
              ELSE
                jx = kx
                DO j = 1, n
                  temp = x(jx)
                  ix = jx
                  IF (nounit) temp = temp*ap(kk)
                  DO k = kk + 1, kk + n - j
                    ix = ix + incx
                    temp = temp + ap(k)*x(ix)
                  END DO
                  x(jx) = temp
                  jx = jx + incx
                  kk = kk + (n-j+1)
                END DO
              END IF
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DTPSV(uplo, trans, diag, n, ap, x, incx)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: incx, n
          CHARACTER :: diag, trans, uplo
          REAL(rp_) :: ap(*), x(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          REAL(rp_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, k, kk, kx
          LOGICAL :: nounit
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          info = 0
          IF (.NOT. LSAME(uplo,'U') .AND. .NOT. LSAME(uplo,'L')) THEN
            info = 1
          ELSE IF (.NOT. LSAME(trans,'N') .AND. .NOT. LSAME(trans,& 
            'T') .AND. .NOT. LSAME(trans,'C')) THEN
            info = 2
          ELSE IF (.NOT. LSAME(diag,'U') .AND. .NOT. LSAME(diag,'N')) &
            THEN
            info = 3
          ELSE IF (n<0) THEN
            info = 4
          ELSE IF (incx==0) THEN
            info = 7
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('TPSV', info)
            RETURN
          END IF
          IF (n==0) RETURN
          nounit = LSAME(diag, 'N')
          IF (incx<=0) THEN
            kx = 1 - (n-1)*incx
          ELSE IF (incx/=1) THEN
            kx = 1
          END IF
          IF (LSAME(trans,'N')) THEN
            IF (LSAME(uplo,'U')) THEN
              kk = (n*(n+1))/2
              IF (incx==1) THEN
                DO j = n, 1_ip_, -1_ip_
                  IF (x(j)/=zero) THEN
                    IF (nounit) x(j) = x(j)/ap(kk)
                    temp = x(j)
                    k = kk - 1
                    DO i = j - 1, 1_ip_, -1_ip_
                      x(i) = x(i) - temp*ap(k)
                      k = k - 1
                    END DO
                  END IF
                  kk = kk - j
                END DO
              ELSE
                jx = kx + (n-1)*incx
                DO j = n, 1_ip_, -1_ip_
                  IF (x(jx)/=zero) THEN
                    IF (nounit) x(jx) = x(jx)/ap(kk)
                    temp = x(jx)
                    ix = jx
                    DO k = kk - 1, kk - j + 1, -1_ip_
                      ix = ix - incx
                      x(ix) = x(ix) - temp*ap(k)
                    END DO
                  END IF
                  jx = jx - incx
                  kk = kk - j
                END DO
              END IF
            ELSE
              kk = 1
              IF (incx==1) THEN
                DO j = 1, n
                  IF (x(j)/=zero) THEN
                    IF (nounit) x(j) = x(j)/ap(kk)
                    temp = x(j)
                    k = kk + 1
                    DO i = j + 1, n
                      x(i) = x(i) - temp*ap(k)
                      k = k + 1
                    END DO
                  END IF
                  kk = kk + (n-j+1)
                END DO
              ELSE
                jx = kx
                DO j = 1, n
                  IF (x(jx)/=zero) THEN
                    IF (nounit) x(jx) = x(jx)/ap(kk)
                    temp = x(jx)
                    ix = jx
                    DO k = kk + 1, kk + n - j
                      ix = ix + incx
                      x(ix) = x(ix) - temp*ap(k)
                    END DO
                  END IF
                  jx = jx + incx
                  kk = kk + (n-j+1)
                END DO
              END IF
            END IF
          ELSE
            IF (LSAME(uplo,'U')) THEN
              kk = 1
              IF (incx==1) THEN
                DO j = 1, n
                  temp = x(j)
                  k = kk
                  DO i = 1, j - 1
                    temp = temp - ap(k)*x(i)
                    k = k + 1
                  END DO
                  IF (nounit) temp = temp/ap(kk+j-1)
                  x(j) = temp
                  kk = kk + j
                END DO
              ELSE
                jx = kx
                DO j = 1, n
                  temp = x(jx)
                  ix = kx
                  DO k = kk, kk + j - 2
                    temp = temp - ap(k)*x(ix)
                    ix = ix + incx
                  END DO
                  IF (nounit) temp = temp/ap(kk+j-1)
                  x(jx) = temp
                  jx = jx + incx
                  kk = kk + j
                END DO
              END IF
            ELSE
              kk = (n*(n+1))/2
              IF (incx==1) THEN
                DO j = n, 1_ip_, -1_ip_
                  temp = x(j)
                  k = kk
                  DO i = n, j + 1, -1_ip_
                    temp = temp - ap(k)*x(i)
                    k = k - 1
                  END DO
                  IF (nounit) temp = temp/ap(kk-n+j)
                  x(j) = temp
                  kk = kk - (n-j+1)
                END DO
              ELSE
                kx = kx + (n-1)*incx
                jx = kx
                DO j = n, 1_ip_, -1_ip_
                  temp = x(jx)
                  ix = kx
                  DO k = kk, kk - (n-(j+1)), -1_ip_
                    temp = temp - ap(k)*x(ix)
                    ix = ix - incx
                  END DO
                  IF (nounit) temp = temp/ap(kk-n+j)
                  x(jx) = temp
                  jx = jx - incx
                  kk = kk - (n-j+1)
                END DO
              END IF
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DTRMM(side, uplo, transa, diag, m, n, alpha, a, lda, b,  &
          ldb)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: alpha
          INTEGER(ip_) :: lda, ldb, m, n
          CHARACTER :: diag, side, transa, uplo
          REAL(rp_) :: a(lda, *), b(ldb, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          INTRINSIC :: MAX
          REAL(rp_) :: temp
          INTEGER(ip_) :: i, info, j, k, nrowa
          LOGICAL :: lside, nounit, upper
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          lside = LSAME(side, 'L')
          IF (lside) THEN
            nrowa = m
          ELSE
            nrowa = n
          END IF
          nounit = LSAME(diag, 'N')
          upper = LSAME(uplo, 'U')
          info = 0
          IF ((.NOT. lside) .AND. (.NOT. LSAME(side,'R'))) THEN
            info = 1
          ELSE IF ((.NOT. upper) .AND. (.NOT. LSAME(uplo,'L'))) THEN
            info = 2
          ELSE IF ((.NOT. LSAME(transa,'N')) .AND. (.NOT. &
            LSAME(transa, 'T')) .AND. (.NOT. LSAME(transa,'C'))) THEN
            info = 3
          ELSE IF ((.NOT. LSAME(diag,'U')) .AND. (.NOT. LSAME(diag, &
            'N'))) THEN
            info = 4
          ELSE IF (m<0) THEN
            info = 5
          ELSE IF (n<0) THEN
            info = 6
          ELSE IF (lda<MAX(1,nrowa)) THEN
            info = 9
          ELSE IF (ldb<MAX(1,m)) THEN
            info = 11
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('TRMM', info)
            RETURN
          END IF
          IF (m==0 .OR. n==0) RETURN
          IF (alpha==zero) THEN
            DO j = 1, n
              DO i = 1, m
                b(i, j) = zero
              END DO
            END DO
            RETURN
          END IF
          IF (lside) THEN
            IF (LSAME(transa,'N')) THEN
              IF (upper) THEN
                DO j = 1, n
                  DO k = 1, m
                    IF (b(k,j)/=zero) THEN
                      temp = alpha*b(k, j)
                      DO i = 1, k - 1
                        b(i, j) = b(i, j) + temp*a(i, k)
                      END DO
                      IF (nounit) temp = temp*a(k, k)
                      b(k, j) = temp
                    END IF
                  END DO
                END DO
              ELSE
                DO j = 1, n
                  DO k = m, 1_ip_, -1_ip_
                    IF (b(k,j)/=zero) THEN
                      temp = alpha*b(k, j)
                      b(k, j) = temp
                      IF (nounit) b(k, j) = b(k, j)*a(k, k)
                      DO i = k + 1, m
                        b(i, j) = b(i, j) + temp*a(i, k)
                      END DO
                    END IF
                  END DO
                END DO
              END IF
            ELSE
              IF (upper) THEN
                DO j = 1, n
                  DO i = m, 1_ip_, -1_ip_
                    temp = b(i, j)
                    IF (nounit) temp = temp*a(i, i)
                    DO k = 1, i - 1
                      temp = temp + a(k, i)*b(k, j)
                    END DO
                    b(i, j) = alpha*temp
                  END DO
                END DO
              ELSE
                DO j = 1, n
                  DO i = 1, m
                    temp = b(i, j)
                    IF (nounit) temp = temp*a(i, i)
                    DO k = i + 1, m
                      temp = temp + a(k, i)*b(k, j)
                    END DO
                    b(i, j) = alpha*temp
                  END DO
                END DO
              END IF
            END IF
          ELSE
            IF (LSAME(transa,'N')) THEN
              IF (upper) THEN
                DO j = n, 1_ip_, -1_ip_
                  temp = alpha
                  IF (nounit) temp = temp*a(j, j)
                  DO i = 1, m
                    b(i, j) = temp*b(i, j)
                  END DO
                  DO k = 1, j - 1
                    IF (a(k,j)/=zero) THEN
                      temp = alpha*a(k, j)
                      DO i = 1, m
                        b(i, j) = b(i, j) + temp*b(i, k)
                      END DO
                    END IF
                  END DO
                END DO
              ELSE
                DO j = 1, n
                  temp = alpha
                  IF (nounit) temp = temp*a(j, j)
                  DO i = 1, m
                    b(i, j) = temp*b(i, j)
                  END DO
                  DO k = j + 1, n
                    IF (a(k,j)/=zero) THEN
                      temp = alpha*a(k, j)
                      DO i = 1, m
                        b(i, j) = b(i, j) + temp*b(i, k)
                      END DO
                    END IF
                  END DO
                END DO
              END IF
            ELSE
              IF (upper) THEN
                DO k = 1, n
                  DO j = 1, k - 1
                    IF (a(j,k)/=zero) THEN
                      temp = alpha*a(j, k)
                      DO i = 1, m
                        b(i, j) = b(i, j) + temp*b(i, k)
                      END DO
                    END IF
                  END DO
                  temp = alpha
                  IF (nounit) temp = temp*a(k, k)
                  IF (temp/=one) THEN
                    DO i = 1, m
                      b(i, k) = temp*b(i, k)
                    END DO
                  END IF
                END DO
              ELSE
                DO k = n, 1_ip_, -1_ip_
                  DO j = k + 1, n
                    IF (a(j,k)/=zero) THEN
                      temp = alpha*a(j, k)
                      DO i = 1, m
                        b(i, j) = b(i, j) + temp*b(i, k)
                      END DO
                    END IF
                  END DO
                  temp = alpha
                  IF (nounit) temp = temp*a(k, k)
                  IF (temp/=one) THEN
                    DO i = 1, m
                      b(i, k) = temp*b(i, k)
                    END DO
                  END IF
                END DO
              END IF
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DTRMV(uplo, trans, diag, n, a, lda, x, incx)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: incx, lda, n
          CHARACTER :: diag, trans, uplo
          REAL(rp_) :: a(lda, *), x(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          REAL(rp_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, kx
          LOGICAL :: nounit
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          INTRINSIC :: MAX
          info = 0
          IF (.NOT. LSAME(uplo,'U') .AND. .NOT. LSAME(uplo,'L')) THEN
            info = 1
          ELSE IF (.NOT. LSAME(trans,'N') .AND. .NOT. LSAME(trans,& 
            'T') .AND. .NOT. LSAME(trans,'C')) THEN
            info = 2
          ELSE IF (.NOT. LSAME(diag,'U') .AND. .NOT. LSAME(diag,'N')) &
            THEN
            info = 3
          ELSE IF (n<0) THEN
            info = 4
          ELSE IF (lda<MAX(1,n)) THEN
            info = 6
          ELSE IF (incx==0) THEN
            info = 8
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('TRMV', info)
            RETURN
          END IF
          IF (n==0) RETURN
          nounit = LSAME(diag, 'N')
          IF (incx<=0) THEN
            kx = 1 - (n-1)*incx
          ELSE IF (incx/=1) THEN
            kx = 1
          END IF
          IF (LSAME(trans,'N')) THEN
            IF (LSAME(uplo,'U')) THEN
              IF (incx==1) THEN
                DO j = 1, n
                  IF (x(j)/=zero) THEN
                    temp = x(j)
                    DO i = 1, j - 1
                      x(i) = x(i) + temp*a(i, j)
                    END DO
                    IF (nounit) x(j) = x(j)*a(j, j)
                  END IF
                END DO
              ELSE
                jx = kx
                DO j = 1, n
                  IF (x(jx)/=zero) THEN
                    temp = x(jx)
                    ix = kx
                    DO i = 1, j - 1
                      x(ix) = x(ix) + temp*a(i, j)
                      ix = ix + incx
                    END DO
                    IF (nounit) x(jx) = x(jx)*a(j, j)
                  END IF
                  jx = jx + incx
                END DO
              END IF
            ELSE
              IF (incx==1) THEN
                DO j = n, 1_ip_, -1_ip_
                  IF (x(j)/=zero) THEN
                    temp = x(j)
                    DO i = n, j + 1, -1_ip_
                      x(i) = x(i) + temp*a(i, j)
                    END DO
                    IF (nounit) x(j) = x(j)*a(j, j)
                  END IF
                END DO
              ELSE
                kx = kx + (n-1)*incx
                jx = kx
                DO j = n, 1_ip_, -1_ip_
                  IF (x(jx)/=zero) THEN
                    temp = x(jx)
                    ix = kx
                    DO i = n, j + 1, -1_ip_
                      x(ix) = x(ix) + temp*a(i, j)
                      ix = ix - incx
                    END DO
                    IF (nounit) x(jx) = x(jx)*a(j, j)
                  END IF
                  jx = jx - incx
                END DO
              END IF
            END IF
          ELSE
            IF (LSAME(uplo,'U')) THEN
              IF (incx==1) THEN
                DO j = n, 1_ip_, -1_ip_
                  temp = x(j)
                  IF (nounit) temp = temp*a(j, j)
                  DO i = j - 1, 1_ip_, -1_ip_
                    temp = temp + a(i, j)*x(i)
                  END DO
                  x(j) = temp
                END DO
              ELSE
                jx = kx + (n-1)*incx
                DO j = n, 1_ip_, -1_ip_
                  temp = x(jx)
                  ix = jx
                  IF (nounit) temp = temp*a(j, j)
                  DO i = j - 1, 1_ip_, -1_ip_
                    ix = ix - incx
                    temp = temp + a(i, j)*x(ix)
                  END DO
                  x(jx) = temp
                  jx = jx - incx
                END DO
              END IF
            ELSE
              IF (incx==1) THEN
                DO j = 1, n
                  temp = x(j)
                  IF (nounit) temp = temp*a(j, j)
                  DO i = j + 1, n
                    temp = temp + a(i, j)*x(i)
                  END DO
                  x(j) = temp
                END DO
              ELSE
                jx = kx
                DO j = 1, n
                  temp = x(jx)
                  ix = jx
                  IF (nounit) temp = temp*a(j, j)
                  DO i = j + 1, n
                    ix = ix + incx
                    temp = temp + a(i, j)*x(ix)
                  END DO
                  x(jx) = temp
                  jx = jx + incx
                END DO
              END IF
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DTRSM(side, uplo, transa, diag, m, n, alpha, a, lda, b,  &
          ldb)
          USE BLAS_LAPACK_KINDS_precision
          REAL(rp_) :: alpha
          INTEGER(ip_) :: lda, ldb, m, n
          CHARACTER :: diag, side, transa, uplo
          REAL(rp_) :: a(lda, *), b(ldb, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          INTRINSIC :: MAX
          REAL(rp_) :: temp
          INTEGER(ip_) :: i, info, j, k, nrowa
          LOGICAL :: lside, nounit, upper
          REAL(rp_) :: one, zero
          PARAMETER (one=1.0_rp_, zero=0.0_rp_)
          lside = LSAME(side, 'L')
          IF (lside) THEN
            nrowa = m
          ELSE
            nrowa = n
          END IF
          nounit = LSAME(diag, 'N')
          upper = LSAME(uplo, 'U')
          info = 0
          IF ((.NOT. lside) .AND. (.NOT. LSAME(side,'R'))) THEN
            info = 1
          ELSE IF ((.NOT. upper) .AND. (.NOT. LSAME(uplo,'L'))) THEN
            info = 2
          ELSE IF ((.NOT. LSAME(transa,'N')) .AND. (.NOT. &
            LSAME(transa, 'T')) .AND. (.NOT. LSAME(transa,'C'))) THEN
            info = 3
          ELSE IF ((.NOT. LSAME(diag,'U')) .AND. (.NOT. LSAME(diag, &
            'N'))) THEN
            info = 4
          ELSE IF (m<0) THEN
            info = 5
          ELSE IF (n<0) THEN
            info = 6
          ELSE IF (lda<MAX(1,nrowa)) THEN
            info = 9
          ELSE IF (ldb<MAX(1,m)) THEN
            info = 11
          END IF
          IF (info/=0) THEN
            CALL XERBLA2('TRSM', info)
            RETURN
          END IF
          IF (m==0 .OR. n==0) RETURN
          IF (alpha==zero) THEN
            DO j = 1, n
              DO i = 1, m
                b(i, j) = zero
              END DO
            END DO
            RETURN
          END IF
          IF (lside) THEN
            IF (LSAME(transa,'N')) THEN
              IF (upper) THEN
                DO j = 1, n
                  IF (alpha/=one) THEN
                    DO i = 1, m
                      b(i, j) = alpha*b(i, j)
                    END DO
                  END IF
                  DO k = m, 1_ip_, -1_ip_
                    IF (b(k,j)/=zero) THEN
                      IF (nounit) b(k, j) = b(k, j)/a(k, k)
                      DO i = 1, k - 1
                        b(i, j) = b(i, j) - b(k, j)*a(i, k)
                      END DO
                    END IF
                  END DO
                END DO
              ELSE
                DO j = 1, n
                  IF (alpha/=one) THEN
                    DO i = 1, m
                      b(i, j) = alpha*b(i, j)
                    END DO
                  END IF
                  DO k = 1, m
                    IF (b(k,j)/=zero) THEN
                      IF (nounit) b(k, j) = b(k, j)/a(k, k)
                      DO i = k + 1, m
                        b(i, j) = b(i, j) - b(k, j)*a(i, k)
                      END DO
                    END IF
                  END DO
                END DO
              END IF
            ELSE
              IF (upper) THEN
                DO j = 1, n
                  DO i = 1, m
                    temp = alpha*b(i, j)
                    DO k = 1, i - 1
                      temp = temp - a(k, i)*b(k, j)
                    END DO
                    IF (nounit) temp = temp/a(i, i)
                    b(i, j) = temp
                  END DO
                END DO
              ELSE
                DO j = 1, n
                  DO i = m, 1_ip_, -1_ip_
                    temp = alpha*b(i, j)
                    DO k = i + 1, m
                      temp = temp - a(k, i)*b(k, j)
                    END DO
                    IF (nounit) temp = temp/a(i, i)
                    b(i, j) = temp
                  END DO
                END DO
              END IF
            END IF
          ELSE
            IF (LSAME(transa,'N')) THEN
              IF (upper) THEN
                DO j = 1, n
                  IF (alpha/=one) THEN
                    DO i = 1, m
                      b(i, j) = alpha*b(i, j)
                    END DO
                  END IF
                  DO k = 1, j - 1
                    IF (a(k,j)/=zero) THEN
                      DO i = 1, m
                        b(i, j) = b(i, j) - a(k, j)*b(i, k)
                      END DO
                    END IF
                  END DO
                  IF (nounit) THEN
                    temp = one/a(j, j)
                    DO i = 1, m
                      b(i, j) = temp*b(i, j)
                    END DO
                  END IF
                END DO
              ELSE
                DO j = n, 1_ip_, -1_ip_
                  IF (alpha/=one) THEN
                    DO i = 1, m
                      b(i, j) = alpha*b(i, j)
                    END DO
                  END IF
                  DO k = j + 1, n
                    IF (a(k,j)/=zero) THEN
                      DO i = 1, m
                        b(i, j) = b(i, j) - a(k, j)*b(i, k)
                      END DO
                    END IF
                  END DO
                  IF (nounit) THEN
                    temp = one/a(j, j)
                    DO i = 1, m
                      b(i, j) = temp*b(i, j)
                    END DO
                  END IF
                END DO
              END IF
            ELSE
              IF (upper) THEN
                DO k = n, 1_ip_, -1_ip_
                  IF (nounit) THEN
                    temp = one/a(k, k)
                    DO i = 1, m
                      b(i, k) = temp*b(i, k)
                    END DO
                  END IF
                  DO j = 1, k - 1
                    IF (a(j,k)/=zero) THEN
                      temp = a(j, k)
                      DO i = 1, m
                        b(i, j) = b(i, j) - temp*b(i, k)
                      END DO
                    END IF
                  END DO
                  IF (alpha/=one) THEN
                    DO i = 1, m
                      b(i, k) = alpha*b(i, k)
                    END DO
                  END IF
                END DO
              ELSE
                DO k = 1, n
                  IF (nounit) THEN
                    temp = one/a(k, k)
                    DO i = 1, m
                      b(i, k) = temp*b(i, k)
                    END DO
                  END IF
                  DO j = k + 1, n
                    IF (a(j,k)/=zero) THEN
                      temp = a(j, k)
                      DO i = 1, m
                        b(i, j) = b(i, j) - temp*b(i, k)
                      END DO
                    END IF
                  END DO
                  IF (alpha/=one) THEN
                    DO i = 1, m
                      b(i, k) = alpha*b(i, k)
                    END DO
                  END IF
                END DO
              END IF
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE DTRSV(uplo, trans, diag, n, a, lda, x, incx)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: incx, lda, n
          CHARACTER :: diag, trans, uplo
          REAL(rp_) :: a(lda, *), x(*)
          REAL(rp_) :: zero
          PARAMETER (zero=0.0_rp_)
          REAL(rp_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, kx
          LOGICAL :: nounit
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA2
          INTRINSIC :: MAX
          info = 0
          IF (.NOT. LSAME(uplo,'U') .AND. .NOT. LSAME(uplo,'L')) THEN
            info = 1
          ELSE IF (.NOT. LSAME(trans,'N') .AND. .NOT. LSAME(trans,& 
            'T') .AND. .NOT. LSAME(trans,'C')) THEN
            info = 2
          ELSE IF (.NOT. LSAME(diag,'U') .AND. .NOT. LSAME(diag,'N')) &
            THEN
            info = 3
          ELSE IF (n<0) THEN
            info = 4
          ELSE IF (lda<MAX(1,n)) THEN
            info = 6
          ELSE IF (incx==0) THEN
            info = 8
          END IF
          IF (info/=0) THEN
            CALL XERBLA2( 'TRSV', info)
            RETURN
          END IF
          IF (n==0) RETURN
          nounit = LSAME(diag, 'N')
          IF (incx<=0) THEN
            kx = 1 - (n-1)*incx
          ELSE IF (incx/=1) THEN
            kx = 1
          END IF
          IF (LSAME(trans,'N')) THEN
            IF (LSAME(uplo,'U')) THEN
              IF (incx==1) THEN
                DO j = n, 1_ip_, -1_ip_
                  IF (x(j)/=zero) THEN
                    IF (nounit) x(j) = x(j)/a(j, j)
                    temp = x(j)
                    DO i = j - 1, 1_ip_, -1_ip_
                      x(i) = x(i) - temp*a(i, j)
                    END DO
                  END IF
                END DO
              ELSE
                jx = kx + (n-1)*incx
                DO j = n, 1_ip_, -1_ip_
                  IF (x(jx)/=zero) THEN
                    IF (nounit) x(jx) = x(jx)/a(j, j)
                    temp = x(jx)
                    ix = jx
                    DO i = j - 1, 1_ip_, -1_ip_
                      ix = ix - incx
                      x(ix) = x(ix) - temp*a(i, j)
                    END DO
                  END IF
                  jx = jx - incx
                END DO
              END IF
            ELSE
              IF (incx==1) THEN
                DO j = 1, n
                  IF (x(j)/=zero) THEN
                    IF (nounit) x(j) = x(j)/a(j, j)
                    temp = x(j)
                    DO i = j + 1, n
                      x(i) = x(i) - temp*a(i, j)
                    END DO
                  END IF
                END DO
              ELSE
                jx = kx
                DO j = 1, n
                  IF (x(jx)/=zero) THEN
                    IF (nounit) x(jx) = x(jx)/a(j, j)
                    temp = x(jx)
                    ix = jx
                    DO i = j + 1, n
                      ix = ix + incx
                      x(ix) = x(ix) - temp*a(i, j)
                    END DO
                  END IF
                  jx = jx + incx
                END DO
              END IF
            END IF
          ELSE
            IF (LSAME(uplo,'U')) THEN
              IF (incx==1) THEN
                DO j = 1, n
                  temp = x(j)
                  DO i = 1, j - 1
                    temp = temp - a(i, j)*x(i)
                  END DO
                  IF (nounit) temp = temp/a(j, j)
                  x(j) = temp
                END DO
              ELSE
                jx = kx
                DO j = 1, n
                  temp = x(jx)
                  ix = kx
                  DO i = 1, j - 1
                    temp = temp - a(i, j)*x(ix)
                    ix = ix + incx
                  END DO
                  IF (nounit) temp = temp/a(j, j)
                  x(jx) = temp
                  jx = jx + incx
                END DO
              END IF
            ELSE
              IF (incx==1) THEN
                DO j = n, 1_ip_, -1_ip_
                  temp = x(j)
                  DO i = n, j + 1, -1_ip_
                    temp = temp - a(i, j)*x(i)
                  END DO
                  IF (nounit) temp = temp/a(j, j)
                  x(j) = temp
                END DO
              ELSE
                kx = kx + (n-1)*incx
                jx = kx
                DO j = n, 1_ip_, -1_ip_
                  temp = x(jx)
                  ix = kx
                  DO i = n, j + 1, -1_ip_
                    temp = temp - a(i, j)*x(ix)
                    ix = ix - incx
                  END DO
                  IF (nounit) temp = temp/a(j, j)
                  x(jx) = temp
                  jx = jx - incx
                END DO
              END IF
            END IF
          END IF
          RETURN
        END SUBROUTINE


        INTEGER(ip_) FUNCTION IDAMAX(n, dx, incx)
          USE BLAS_LAPACK_KINDS_precision
          INTEGER(ip_) :: incx, n
          REAL(rp_) :: dx(*)
          REAL(rp_) :: dmax
          INTEGER(ip_) :: i, ix
          INTRINSIC :: ABS
          IDAMAX = 0
          IF (n<1 .OR. incx<=0) RETURN
          IDAMAX = 1
          IF (n==1) RETURN
          IF (incx==1) THEN
            dmax = ABS(dx(1))
            DO i = 2, n
              IF (ABS(dx(i))>dmax) THEN
                IDAMAX = i
                dmax = ABS(dx(i))
              END IF
            END DO
          ELSE
            ix = 1
            dmax = ABS(dx(1))
            ix = ix + incx
            DO i = 2, n
              IF (ABS(dx(ix))>dmax) THEN
                IDAMAX = i
                dmax = ABS(dx(ix))
              END IF
              ix = ix + incx
            END DO
          END IF
          RETURN
        END FUNCTION

        LOGICAL FUNCTION LSAME(ca, cb)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER :: ca, cb
          INTRINSIC :: ICHAR
          INTEGER(ip_) :: inta, intb, zcode
          LSAME = ca == cb
          IF (LSAME) RETURN
          zcode = ICHAR('Z')
          inta = ICHAR(ca)
          intb = ICHAR(cb)
          IF (zcode==90 .OR. zcode==122) THEN
            IF (inta>=97 .AND. inta<=122) inta = inta - 32
            IF (intb>=97 .AND. intb<=122) intb = intb - 32
          ELSE IF (zcode==233 .OR. zcode==169) THEN
            IF (inta>=129 .AND. inta<=137 .OR. inta>=145 .AND. inta<=153    &
              .OR. inta>=162 .AND. inta<=169) inta = inta + 64
            IF (intb>=129 .AND. intb<=137 .OR. intb>=145 .AND. intb<=153    &
              .OR. intb>=162 .AND. intb<=169) intb = intb + 64
          ELSE IF (zcode==218 .OR. zcode==250) THEN
            IF (inta>=225 .AND. inta<=250) inta = inta - 32
            IF (intb>=225 .AND. intb<=250) intb = intb - 32
          END IF
          LSAME = inta == intb
        END FUNCTION

!  modified version of XERBLA for which the base_name excludes the
!  first (precision identifying) character.

        SUBROUTINE XERBLA2(base_srname, info)
          USE BLAS_LAPACK_KINDS_precision
          CHARACTER (*) :: base_srname
          INTEGER(ip_) :: info
          INTRINSIC :: LEN_TRIM
!  added for BLAS/LAPACK templating
          CHARACTER ( LEN = LEN( base_srname ) + 4 ) :: srname
          CHARACTER ( LEN = 1 ) :: real_prefix
          CHARACTER ( LEN = 3 ) :: int_suffix
          IF ( rp_ == REAL32 ) THEN
            real_prefix = 'S'
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
          srname = real_prefix // base_srname // int_suffix

          WRITE (*, FMT=9999) TRIM(srname), info
          STOP
 9999     FORMAT (' ** On entry to ', A, ' parameter number ', I0, &
                  ' had an illegal value')
        END SUBROUTINE
