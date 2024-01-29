! THIS VERSION: GALAHAD 4.3 - 2024-01-29 AT 13:26 GMT

#include "galahad_blas.h"

! Reference blas, see http://www.netlib.org/lapack/explore-html/

        REAL(r8_) FUNCTION DASUM(n, dx, incx)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, n
          REAL(r8_) :: dx(*)
          REAL(r8_) :: dtemp
          INTEGER(ip_) :: i, m, mp1, nincx
          INTRINSIC :: DABS, MOD
          DASUM = 0.0_r8_
          dtemp = 0.0_r8_
          IF (n<=0 .OR. incx<=0) RETURN
          IF (incx==1) THEN
            m = MOD(n, 6)
            IF (m/=0) THEN
              DO i = 1, m
                dtemp = dtemp + DABS(dx(i))
              END DO
              IF (n<6) THEN
                DASUM = dtemp
                RETURN
              END IF
            END IF
            mp1 = m + 1
            DO i = mp1, n, 6
              dtemp = dtemp + DABS(dx(i)) + DABS(dx(i+1)) + DABS(dx(i+2))   &
                + DABS(dx(i+3)) + DABS(dx(i+4)) + DABS(dx(i+5))
            END DO
          ELSE
            nincx = n*incx
            DO i = 1, nincx, incx
              dtemp = dtemp + DABS(dx(i))
            END DO
          END IF
          DASUM = dtemp
          RETURN
        END FUNCTION

        SUBROUTINE DAXPY(n, da, dx, incx, dy, incy)
          USE GALAHAD_KINDS
          REAL(r8_) :: da
          INTEGER(ip_) :: incx, incy, n
          REAL(r8_) :: dx(*), dy(*)
          INTEGER(ip_) :: i, ix, iy, m, mp1
          INTRINSIC :: MOD
          IF (n<=0) RETURN
          IF (da==0.0_r8_) RETURN
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

        REAL(r8_) FUNCTION DCABS1(z)
          USE GALAHAD_KINDS
          COMPLEX(c8_) :: z
          INTRINSIC :: ABS, DBLE, DIMAG
          DCABS1 = ABS(DBLE(z)) + ABS(DIMAG(z))
          RETURN
        END FUNCTION

        SUBROUTINE DCOPY(n, dx, incx, dy, incy)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, incy, n
          REAL(r8_) :: dx(*), dy(*)
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

        REAL(r8_) FUNCTION DDOT(n, dx, incx, dy, incy)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, incy, n
          REAL(r8_) :: dx(*), dy(*)
          REAL(r8_) :: dtemp
          INTEGER(ip_) :: i, ix, iy, m, mp1
          INTRINSIC :: MOD
          DDOT = 0.0_r8_
          dtemp = 0.0_r8_
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
          USE GALAHAD_KINDS
          REAL(r8_) :: alpha, beta
          INTEGER(ip_) :: k, lda, ldb, ldc, m, n
          CHARACTER :: transa, transb
          REAL(r8_) :: a(lda, *), b(ldb, *), c(ldc, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: MAX
          REAL(r8_) :: temp
          INTEGER(ip_) :: i, info, j, l, nrowa, nrowb
          LOGICAL :: nota, notb
          REAL(r8_) :: one, zero
          PARAMETER (one=1.0_r8_, zero=0.0_r8_)
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
            CALL XERBLA('DGEMM ', info)
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
          USE GALAHAD_KINDS
          REAL(r8_) :: alpha, beta
          INTEGER(ip_) :: incx, incy, lda, m, n
          CHARACTER :: trans
          REAL(r8_) :: a(lda, *), x(*), y(*)
          REAL(r8_) :: one, zero
          PARAMETER (one=1.0_r8_, zero=0.0_r8_)
          REAL(r8_) :: temp
          INTEGER(ip_) :: i, info, ix, iy, j, jx, jy, kx, ky, lenx, leny
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('DGEMV ', info)
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
          USE GALAHAD_KINDS
          REAL(r8_) :: alpha
          INTEGER(ip_) :: incx, incy, lda, m, n
          REAL(r8_) :: a(lda, *), x(*), y(*)
          REAL(r8_) :: zero
          PARAMETER (zero=0.0_r8_)
          REAL(r8_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jy, kx
          EXTERNAL :: XERBLA
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
            CALL XERBLA('DGER  ', info)
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

        REAL(r8_) FUNCTION DNRM2(n, x, incx)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, n
          REAL(r8_) :: x(*)
          REAL(r8_) :: one, zero
          PARAMETER (one=1.0_r8_, zero=0.0_r8_)
          REAL(r8_) :: absxi, norm, scale, ssq
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
          USE GALAHAD_KINDS
          REAL(r8_) :: c, s
          INTEGER(ip_) :: incx, incy, n
          REAL(r8_) :: dx(*), dy(*)
          REAL(r8_) :: dtemp
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
          USE GALAHAD_KINDS
          REAL(r8_) :: c, da, db, s
          REAL(r8_) :: r, roe, scale, z
          INTRINSIC :: DABS, DSIGN, DSQRT
          scale = DABS(da) + DABS(db)
          IF (scale==0.0_r8_) THEN
            c = 1.0_r8_
            s = 0.0_r8_
            r = 0.0_r8_
            z = 0.0_r8_
          ELSE
            roe = db
            IF (DABS(da)>DABS(db)) roe = da
            r = scale*DSQRT((da/scale)**2+(db/scale)**2)
            r = DSIGN(1.0_r8_, roe)*r
            c = da/r
            s = db/r
            z = 1.0_r8_
            IF (DABS(da)>DABS(db)) z = s
            IF (DABS(db)>=DABS(da) .AND. c/=0.0_r8_) z = 1.0_r8_/c
          END IF
          da = r
          db = z
          RETURN
        END SUBROUTINE

        SUBROUTINE DSCAL(n, da, dx, incx)
          USE GALAHAD_KINDS
          REAL(r8_) :: da
          INTEGER(ip_) :: incx, n
          REAL(r8_) :: dx(*)
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

        SUBROUTINE DSWAP(n, dx, incx, dy, incy)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, incy, n
          REAL(r8_) :: dx(*), dy(*)
          REAL(r8_) :: dtemp
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
          USE GALAHAD_KINDS
          REAL(r8_) :: alpha, beta
          INTEGER(ip_) :: lda, ldb, ldc, m, n
          CHARACTER :: side, uplo
          REAL(r8_) :: a(lda, *), b(ldb, *), c(ldc, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: MAX
          REAL(r8_) :: temp1, temp2
          INTEGER(ip_) :: i, info, j, k, nrowa
          LOGICAL :: upper
          REAL(r8_) :: one, zero
          PARAMETER (one=1.0_r8_, zero=0.0_r8_)
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
            CALL XERBLA('DSYMM ', info)
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
          USE GALAHAD_KINDS
          REAL(r8_) :: alpha, beta
          INTEGER(ip_) :: incx, incy, lda, n
          CHARACTER :: uplo
          REAL(r8_) :: a(lda, *), x(*), y(*)
          REAL(r8_) :: one, zero
          PARAMETER (one=1.0_r8_, zero=0.0_r8_)
          REAL(r8_) :: temp1, temp2
          INTEGER(ip_) :: i, info, ix, iy, j, jx, jy, kx, ky
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('DSYMV ', info)
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
          USE GALAHAD_KINDS
          REAL(r8_) :: alpha
          INTEGER(ip_) :: incx, lda, n
          CHARACTER :: uplo
          REAL(r8_) :: a(lda, *), x(*)
          REAL(r8_) :: zero
          PARAMETER (zero=0.0_r8_)
          REAL(r8_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, kx
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('DSYR  ', info)
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
          USE GALAHAD_KINDS
          REAL(r8_) :: alpha
          INTEGER(ip_) :: incx, incy, lda, n
          CHARACTER :: uplo
          REAL(r8_) :: a(lda, *), x(*), y(*)
          REAL(r8_) :: zero
          PARAMETER (zero=0.0_r8_)
          REAL(r8_) :: temp1, temp2
          INTEGER(ip_) :: i, info, ix, iy, j, jx, jy, kx, ky
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('DSYR2 ', info)
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
          USE GALAHAD_KINDS
          REAL(r8_) :: alpha, beta
          INTEGER(ip_) :: k, lda, ldb, ldc, n
          CHARACTER :: trans, uplo
          REAL(r8_) :: a(lda, *), b(ldb, *), c(ldc, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: MAX
          REAL(r8_) :: temp1, temp2
          INTEGER(ip_) :: i, info, j, l, nrowa
          LOGICAL :: upper
          REAL(r8_) :: one, zero
          PARAMETER (one=1.0_r8_, zero=0.0_r8_)
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
            CALL XERBLA('DSYR2K', info)
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
          USE GALAHAD_KINDS
          REAL(r8_) :: alpha, beta
          INTEGER(ip_) :: k, lda, ldc, n
          CHARACTER :: trans, uplo
          REAL(r8_) :: a(lda, *), c(ldc, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: MAX
          REAL(r8_) :: temp
          INTEGER(ip_) :: i, info, j, l, nrowa
          LOGICAL :: upper
          REAL(r8_) :: one, zero
          PARAMETER (one=1.0_r8_, zero=0.0_r8_)
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
            CALL XERBLA('DSYRK ', info)
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
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, k, lda, n
          CHARACTER :: diag, trans, uplo
          REAL(r8_) :: a(lda, *), x(*)
          REAL(r8_) :: zero
          PARAMETER (zero=0.0_r8_)
          REAL(r8_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, kplus1, kx, l
          LOGICAL :: nounit
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('DTBSV ', info)
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
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, n
          CHARACTER :: diag, trans, uplo
          REAL(r8_) :: ap(*), x(*)
          REAL(r8_) :: zero
          PARAMETER (zero=0.0_r8_)
          REAL(r8_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, k, kk, kx
          LOGICAL :: nounit
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('DTPMV ', info)
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
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, n
          CHARACTER :: diag, trans, uplo
          REAL(r8_) :: ap(*), x(*)
          REAL(r8_) :: zero
          PARAMETER (zero=0.0_r8_)
          REAL(r8_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, k, kk, kx
          LOGICAL :: nounit
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('DTPSV ', info)
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
          USE GALAHAD_KINDS
          REAL(r8_) :: alpha
          INTEGER(ip_) :: lda, ldb, m, n
          CHARACTER :: diag, side, transa, uplo
          REAL(r8_) :: a(lda, *), b(ldb, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: MAX
          REAL(r8_) :: temp
          INTEGER(ip_) :: i, info, j, k, nrowa
          LOGICAL :: lside, nounit, upper
          REAL(r8_) :: one, zero
          PARAMETER (one=1.0_r8_, zero=0.0_r8_)
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
            CALL XERBLA('DTRMM ', info)
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
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, lda, n
          CHARACTER :: diag, trans, uplo
          REAL(r8_) :: a(lda, *), x(*)
          REAL(r8_) :: zero
          PARAMETER (zero=0.0_r8_)
          REAL(r8_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, kx
          LOGICAL :: nounit
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('DTRMV ', info)
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
          USE GALAHAD_KINDS
          REAL(r8_) :: alpha
          INTEGER(ip_) :: lda, ldb, m, n
          CHARACTER :: diag, side, transa, uplo
          REAL(r8_) :: a(lda, *), b(ldb, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: MAX
          REAL(r8_) :: temp
          INTEGER(ip_) :: i, info, j, k, nrowa
          LOGICAL :: lside, nounit, upper
          REAL(r8_) :: one, zero
          PARAMETER (one=1.0_r8_, zero=0.0_r8_)
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
            CALL XERBLA('DTRSM ', info)
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
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, lda, n
          CHARACTER :: diag, trans, uplo
          REAL(r8_) :: a(lda, *), x(*)
          REAL(r8_) :: zero
          PARAMETER (zero=0.0_r8_)
          REAL(r8_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, kx
          LOGICAL :: nounit
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('DTRSV ', info)
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

        REAL(r8_) FUNCTION DZNRM2(n, x, incx)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, n
          COMPLEX(c8_) :: x(*)
          REAL(r8_) :: one, zero
          PARAMETER (one=1.0_r8_, zero=0.0_r8_)
          REAL(r8_) :: norm, scale, ssq, temp
          INTEGER(ip_) :: ix
          INTRINSIC :: ABS, DBLE, DIMAG, SQRT
          IF (n<1 .OR. incx<1) THEN
            norm = zero
          ELSE
            scale = zero
            ssq = one
            DO ix = 1, 1 + (n-1)*incx, incx
              IF (DBLE(x(ix))/=zero) THEN
                temp = ABS(DBLE(x(ix)))
                IF (scale<temp) THEN
                  ssq = one + ssq*(scale/temp)**2
                  scale = temp
                ELSE
                  ssq = ssq + (temp/scale)**2
                END IF
              END IF
              IF (DIMAG(x(ix))/=zero) THEN
                temp = ABS(DIMAG(x(ix)))
                IF (scale<temp) THEN
                  ssq = one + ssq*(scale/temp)**2
                  scale = temp
                ELSE
                  ssq = ssq + (temp/scale)**2
                END IF
              END IF
            END DO
            norm = scale*SQRT(ssq)
          END IF
          DZNRM2 = norm
          RETURN
        END FUNCTION

        INTEGER(ip_) FUNCTION IDAMAX(n, dx, incx)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, n
          REAL(r8_) :: dx(*)
          REAL(r8_) :: dmax
          INTEGER(ip_) :: i, ix
          INTRINSIC :: DABS
          IDAMAX = 0
          IF (n<1 .OR. incx<=0) RETURN
          IDAMAX = 1
          IF (n==1) RETURN
          IF (incx==1) THEN
            dmax = DABS(dx(1))
            DO i = 2, n
              IF (DABS(dx(i))>dmax) THEN
                IDAMAX = i
                dmax = DABS(dx(i))
              END IF
            END DO
          ELSE
            ix = 1
            dmax = DABS(dx(1))
            ix = ix + incx
            DO i = 2, n
              IF (DABS(dx(ix))>dmax) THEN
                IDAMAX = i
                dmax = DABS(dx(ix))
              END IF
              ix = ix + incx
            END DO
          END IF
          RETURN
        END FUNCTION

        INTEGER(ip_) FUNCTION ISAMAX(n, sx, incx)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, n
          REAL(r4_) :: sx(*)
          REAL(r4_) :: smax
          INTEGER(ip_) :: i, ix
          INTRINSIC :: ABS
          ISAMAX = 0
          IF (n<1 .OR. incx<=0) RETURN
          ISAMAX = 1
          IF (n==1) RETURN
          IF (incx==1) THEN
            smax = ABS(sx(1))
            DO i = 2, n
              IF (ABS(sx(i))>smax) THEN
                ISAMAX = i
                smax = ABS(sx(i))
              END IF
            END DO
          ELSE
            ix = 1
            smax = ABS(sx(1))
            ix = ix + incx
            DO i = 2, n
              IF (ABS(sx(ix))>smax) THEN
                ISAMAX = i
                smax = ABS(sx(ix))
              END IF
              ix = ix + incx
            END DO
          END IF
          RETURN
        END FUNCTION

        LOGICAL FUNCTION LSAME(ca, cb)
          USE GALAHAD_KINDS
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

        REAL(r4_) FUNCTION SASUM(n, sx, incx)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, n
          REAL(r4_) :: sx(*)
          REAL(r4_) :: stemp
          INTEGER(ip_) :: i, m, mp1, nincx
          INTRINSIC :: ABS, MOD
          SASUM = 0.0_r4_
          stemp = 0.0_r4_
          IF (n<=0 .OR. incx<=0) RETURN
          IF (incx==1) THEN
            m = MOD(n, 6)
            IF (m/=0) THEN
              DO i = 1, m
                stemp = stemp + ABS(sx(i))
              END DO
              IF (n<6) THEN
                SASUM = stemp
                RETURN
              END IF
            END IF
            mp1 = m + 1
            DO i = mp1, n, 6
              stemp = stemp + ABS(sx(i)) + ABS(sx(i+1)) + ABS(sx(i+2)) +    &
                ABS(sx(i+3)) + ABS(sx(i+4)) + ABS(sx(i+5))
            END DO
          ELSE
            nincx = n*incx
            DO i = 1, nincx, incx
              stemp = stemp + ABS(sx(i))
            END DO
          END IF
          SASUM = stemp
          RETURN
        END FUNCTION

        SUBROUTINE SAXPY(n, sa, sx, incx, sy, incy)
          USE GALAHAD_KINDS
          REAL(r4_) :: sa
          INTEGER(ip_) :: incx, incy, n
          REAL(r4_) :: sx(*), sy(*)
          INTEGER(ip_) :: i, ix, iy, m, mp1
          INTRINSIC :: MOD
          IF (n<=0) RETURN
          IF (sa==0.0) RETURN
          IF (incx==1 .AND. incy==1) THEN
            m = MOD(n, 4_ip_)
            IF (m/=0) THEN
              DO i = 1, m
                sy(i) = sy(i) + sa*sx(i)
              END DO
            END IF
            IF (n<4) RETURN
            mp1 = m + 1
            DO i = mp1, n, 4
              sy(i) = sy(i) + sa*sx(i)
              sy(i+1) = sy(i+1) + sa*sx(i+1)
              sy(i+2) = sy(i+2) + sa*sx(i+2)
              sy(i+3) = sy(i+3) + sa*sx(i+3)
            END DO
          ELSE
            ix = 1
            iy = 1
            IF (incx<0) ix = (-n+1)*incx + 1
            IF (incy<0) iy = (-n+1)*incy + 1
            DO i = 1, n
              sy(iy) = sy(iy) + sa*sx(ix)
              ix = ix + incx
              iy = iy + incy
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE SCOPY(n, sx, incx, sy, incy)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, incy, n
          REAL(r4_) :: sx(*), sy(*)
          INTEGER(ip_) :: i, ix, iy, m, mp1
          INTRINSIC :: MOD
          IF (n<=0) RETURN
          IF (incx==1 .AND. incy==1) THEN
            m = MOD(n, 7)
            IF (m/=0) THEN
              DO i = 1, m
                sy(i) = sx(i)
              END DO
              IF (n<7) RETURN
            END IF
            mp1 = m + 1
            DO i = mp1, n, 7
              sy(i) = sx(i)
              sy(i+1) = sx(i+1)
              sy(i+2) = sx(i+2)
              sy(i+3) = sx(i+3)
              sy(i+4) = sx(i+4)
              sy(i+5) = sx(i+5)
              sy(i+6) = sx(i+6)
            END DO
          ELSE
            ix = 1
            iy = 1
            IF (incx<0) ix = (-n+1)*incx + 1
            IF (incy<0) iy = (-n+1)*incy + 1
            DO i = 1, n
              sy(iy) = sx(ix)
              ix = ix + incx
              iy = iy + incy
            END DO
          END IF
          RETURN
        END SUBROUTINE

        REAL(r4_) FUNCTION SDOT(n, sx, incx, sy, incy)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, incy, n
          REAL(r4_) :: sx(*), sy(*)
          REAL(r4_) :: stemp
          INTEGER(ip_) :: i, ix, iy, m, mp1
          INTRINSIC :: MOD
          stemp = 0.0_r4_
          SDOT = 0.0_r4_
          IF (n<=0) RETURN
          IF (incx==1 .AND. incy==1) THEN
            m = MOD(n, 5)
            IF (m/=0) THEN
              DO i = 1, m
                stemp = stemp + sx(i)*sy(i)
              END DO
              IF (n<5) THEN
                SDOT = stemp
                RETURN
              END IF
            END IF
            mp1 = m + 1
            DO i = mp1, n, 5
              stemp = stemp + sx(i)*sy(i) + sx(i+1)*sy(i+1) +               &
                sx(i+2)*sy(i+2) + sx(i+3)*sy(i+3) + sx(i+4)*sy(i+4)
            END DO
          ELSE
            ix = 1
            iy = 1
            IF (incx<0) ix = (-n+1)*incx + 1
            IF (incy<0) iy = (-n+1)*incy + 1
            DO i = 1, n
              stemp = stemp + sx(ix)*sy(iy)
              ix = ix + incx
              iy = iy + incy
            END DO
          END IF
          SDOT = stemp
          RETURN
        END FUNCTION

        SUBROUTINE SGEMM(transa, transb, m, n, k, alpha, a, lda, b, ldb,    &
          beta, c, ldc)
          USE GALAHAD_KINDS
          REAL(r4_) :: alpha, beta
          INTEGER(ip_) :: k, lda, ldb, ldc, m, n
          CHARACTER :: transa, transb
          REAL(r4_) :: a(lda, *), b(ldb, *), c(ldc, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: MAX
          REAL(r4_) :: temp
          INTEGER(ip_) :: i, info, j, l, nrowa, nrowb
          LOGICAL :: nota, notb
          REAL(r4_) :: one, zero
          PARAMETER (one=1.0_r4_, zero=0.0_r4_)
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
            CALL XERBLA('SGEMM ', info)
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

        SUBROUTINE SGEMV(trans, m, n, alpha, a, lda, x, incx, beta, y,      &
          incy)
          USE GALAHAD_KINDS
          REAL(r4_) :: alpha, beta
          INTEGER(ip_) :: incx, incy, lda, m, n
          CHARACTER :: trans
          REAL(r4_) :: a(lda, *), x(*), y(*)
          REAL(r4_) :: one, zero
          PARAMETER (one=1.0_r4_, zero=0.0_r4_)
          REAL(r4_) :: temp
          INTEGER(ip_) :: i, info, ix, iy, j, jx, jy, kx, ky, lenx, leny
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('SGEMV ', info)
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

        SUBROUTINE SGER(m, n, alpha, x, incx, y, incy, a, lda)
          USE GALAHAD_KINDS
          REAL(r4_) :: alpha
          INTEGER(ip_) :: incx, incy, lda, m, n
          REAL(r4_) :: a(lda, *), x(*), y(*)
          REAL(r4_) :: zero
          PARAMETER (zero=0.0_r4_)
          REAL(r4_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jy, kx
          EXTERNAL :: XERBLA
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
            CALL XERBLA('SGER  ', info)
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

        REAL(r4_) FUNCTION SNRM2(n, x, incx)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, n
          REAL(r4_) :: x(*)
          REAL(r4_) :: one, zero
          PARAMETER (one=1.0_r4_, zero=0.0_r4_)
          REAL(r4_) :: absxi, norm, scale, ssq
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
          SNRM2 = norm
          RETURN
        END FUNCTION

        SUBROUTINE SROT(n, sx, incx, sy, incy, c, s)
          USE GALAHAD_KINDS
          REAL(r4_) :: c, s
          INTEGER(ip_) :: incx, incy, n
          REAL(r4_) :: sx(*), sy(*)
          REAL(r4_) :: stemp
          INTEGER(ip_) :: i, ix, iy
          IF (n<=0) RETURN
          IF (incx==1 .AND. incy==1) THEN
            DO i = 1, n
              stemp = c*sx(i) + s*sy(i)
              sy(i) = c*sy(i) - s*sx(i)
              sx(i) = stemp
            END DO
          ELSE
            ix = 1
            iy = 1
            IF (incx<0) ix = (-n+1)*incx + 1
            IF (incy<0) iy = (-n+1)*incy + 1
            DO i = 1, n
              stemp = c*sx(ix) + s*sy(iy)
              sy(iy) = c*sy(iy) - s*sx(ix)
              sx(ix) = stemp
              ix = ix + incx
              iy = iy + incy
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE SROTG(sa, sb, c, s)
          USE GALAHAD_KINDS
          REAL(r4_) :: c, s, sa, sb
          REAL(r4_) :: r, roe, scale, z
          INTRINSIC :: ABS, SIGN, SQRT
          scale = ABS(sa) + ABS(sb)
          IF (scale==0.0) THEN
            c = 1.0
            s = 0.0
            r = 0.0
            z = 0.0
          ELSE
            roe = sb
            IF (ABS(sa)>ABS(sb)) roe = sa
            r = scale*SQRT((sa/scale)**2+(sb/scale)**2)
            r = SIGN(1.0, roe)*r
            c = sa/r
            s = sb/r
            z = 1.0
            IF (ABS(sa)>ABS(sb)) z = s
            IF (ABS(sb)>=ABS(sa) .AND. c/=0.0) z = 1.0/c
          END IF
          sa = r
          sb = z
          RETURN
        END SUBROUTINE

        SUBROUTINE SSCAL(n, sa, sx, incx)
          USE GALAHAD_KINDS
          REAL(r4_) :: sa
          INTEGER(ip_) :: incx, n
          REAL(r4_) :: sx(*)
          INTEGER(ip_) :: i, m, mp1, nincx
          INTRINSIC :: MOD
          IF (n<=0 .OR. incx<=0) RETURN
          IF (incx==1) THEN
            m = MOD(n, 5)
            IF (m/=0) THEN
              DO i = 1, m
                sx(i) = sa*sx(i)
              END DO
              IF (n<5) RETURN
            END IF
            mp1 = m + 1
            DO i = mp1, n, 5
              sx(i) = sa*sx(i)
              sx(i+1) = sa*sx(i+1)
              sx(i+2) = sa*sx(i+2)
              sx(i+3) = sa*sx(i+3)
              sx(i+4) = sa*sx(i+4)
            END DO
          ELSE
            nincx = n*incx
            DO i = 1, nincx, incx
              sx(i) = sa*sx(i)
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE SSWAP(n, sx, incx, sy, incy)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, incy, n
          REAL(r4_) :: sx(*), sy(*)
          REAL(r4_) :: stemp
          INTEGER(ip_) :: i, ix, iy, m, mp1
          INTRINSIC :: MOD
          IF (n<=0) RETURN
          IF (incx==1 .AND. incy==1) THEN
            m = MOD(n, 3_ip_)
            IF (m/=0) THEN
              DO i = 1, m
                stemp = sx(i)
                sx(i) = sy(i)
                sy(i) = stemp
              END DO
              IF (n<3) RETURN
            END IF
            mp1 = m + 1
            DO i = mp1, n, 3
              stemp = sx(i)
              sx(i) = sy(i)
              sy(i) = stemp
              stemp = sx(i+1)
              sx(i+1) = sy(i+1)
              sy(i+1) = stemp
              stemp = sx(i+2)
              sx(i+2) = sy(i+2)
              sy(i+2) = stemp
            END DO
          ELSE
            ix = 1
            iy = 1
            IF (incx<0) ix = (-n+1)*incx + 1
            IF (incy<0) iy = (-n+1)*incy + 1
            DO i = 1, n
              stemp = sx(ix)
              sx(ix) = sy(iy)
              sy(iy) = stemp
              ix = ix + incx
              iy = iy + incy
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE SSYMM(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c,  &
          ldc)
          USE GALAHAD_KINDS
          REAL(r4_) :: alpha, beta
          INTEGER(ip_) :: lda, ldb, ldc, m, n
          CHARACTER :: side, uplo
          REAL(r4_) :: a(lda, *), b(ldb, *), c(ldc, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: MAX
          REAL(r4_) :: temp1, temp2
          INTEGER(ip_) :: i, info, j, k, nrowa
          LOGICAL :: upper
          REAL(r4_) :: one, zero
          PARAMETER (one=1.0_r4_, zero=0.0_r4_)
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
            CALL XERBLA('SSYMM ', info)
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

        SUBROUTINE SSYMV(uplo, n, alpha, a, lda, x, incx, beta, y, incy)
          USE GALAHAD_KINDS
          REAL(r4_) :: alpha, beta
          INTEGER(ip_) :: incx, incy, lda, n
          CHARACTER :: uplo
          REAL(r4_) :: a(lda, *), x(*), y(*)
          REAL(r4_) :: one, zero
          PARAMETER (one=1.0_r4_, zero=0.0_r4_)
          REAL(r4_) :: temp1, temp2
          INTEGER(ip_) :: i, info, ix, iy, j, jx, jy, kx, ky
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('SSYMV ', info)
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

        SUBROUTINE SSYR(uplo, n, alpha, x, incx, a, lda)
          USE GALAHAD_KINDS
          REAL(r4_) :: alpha
          INTEGER(ip_) :: incx, lda, n
          CHARACTER :: uplo
          REAL(r4_) :: a(lda, *), x(*)
          REAL(r4_) :: zero
          PARAMETER (zero=0.0_r4_)
          REAL(r4_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, kx
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('SSYR  ', info)
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

        SUBROUTINE SSYR2(uplo, n, alpha, x, incx, y, incy, a, lda)
          USE GALAHAD_KINDS
          REAL(r4_) :: alpha
          INTEGER(ip_) :: incx, incy, lda, n
          CHARACTER :: uplo
          REAL(r4_) :: a(lda, *), x(*), y(*)
          REAL(r4_) :: zero
          PARAMETER (zero=0.0_r4_)
          REAL(r4_) :: temp1, temp2
          INTEGER(ip_) :: i, info, ix, iy, j, jx, jy, kx, ky
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('SSYR2 ', info)
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

        SUBROUTINE SSYR2K(uplo, trans, n, k, alpha, a, lda, b, ldb, beta,   &
          c, ldc)
          USE GALAHAD_KINDS
          REAL(r4_) :: alpha, beta
          INTEGER(ip_) :: k, lda, ldb, ldc, n
          CHARACTER :: trans, uplo
          REAL(r4_) :: a(lda, *), b(ldb, *), c(ldc, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: MAX
          REAL(r4_) :: temp1, temp2
          INTEGER(ip_) :: i, info, j, l, nrowa
          LOGICAL :: upper
          REAL(r4_) :: one, zero
          PARAMETER (one=1.0_r4_, zero=0.0_r4_)
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
            CALL XERBLA('SSYR2K', info)
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

        SUBROUTINE SSYRK(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
          USE GALAHAD_KINDS
          REAL(r4_) :: alpha, beta
          INTEGER(ip_) :: k, lda, ldc, n
          CHARACTER :: trans, uplo
          REAL(r4_) :: a(lda, *), c(ldc, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: MAX
          REAL(r4_) :: temp
          INTEGER(ip_) :: i, info, j, l, nrowa
          LOGICAL :: upper
          REAL(r4_) :: one, zero
          PARAMETER (one=1.0_r4_, zero=0.0_r4_)
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
            CALL XERBLA('SSYRK ', info)
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

        SUBROUTINE STBSV(uplo, trans, diag, n, k, a, lda, x, incx)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, k, lda, n
          CHARACTER :: diag, trans, uplo
          REAL(r4_) :: a(lda, *), x(*)
          REAL(r4_) :: zero
          PARAMETER (zero=0.0_r4_)
          REAL(r4_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, kplus1, kx, l
          LOGICAL :: nounit
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('STBSV ', info)
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

        SUBROUTINE STPMV(uplo, trans, diag, n, ap, x, incx)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, n
          CHARACTER :: diag, trans, uplo
          REAL(r4_) :: ap(*), x(*)
          REAL(r4_) :: zero
          PARAMETER (zero=0.0_r4_)
          REAL(r4_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, k, kk, kx
          LOGICAL :: nounit
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('STPMV ', info)
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

        SUBROUTINE STPSV(uplo, trans, diag, n, ap, x, incx)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, n
          CHARACTER :: diag, trans, uplo
          REAL(r4_) :: ap(*), x(*)
          REAL(r4_) :: zero
          PARAMETER (zero=0.0_r4_)
          REAL(r4_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, k, kk, kx
          LOGICAL :: nounit
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('STPSV ', info)
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

        SUBROUTINE STRMM(side, uplo, transa, diag, m, n, alpha, a, lda, b,  &
          ldb)
          USE GALAHAD_KINDS
          REAL(r4_) :: alpha
          INTEGER(ip_) :: lda, ldb, m, n
          CHARACTER :: diag, side, transa, uplo
          REAL(r4_) :: a(lda, *), b(ldb, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: MAX
          REAL(r4_) :: temp
          INTEGER(ip_) :: i, info, j, k, nrowa
          LOGICAL :: lside, nounit, upper
          REAL(r4_) :: one, zero
          PARAMETER (one=1.0_r4_, zero=0.0_r4_)
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
            CALL XERBLA('STRMM ', info)
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

        SUBROUTINE STRMV(uplo, trans, diag, n, a, lda, x, incx)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, lda, n
          CHARACTER :: diag, trans, uplo
          REAL(r4_) :: a(lda, *), x(*)
          REAL(r4_) :: zero
          PARAMETER (zero=0.0_r4_)
          REAL(r4_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, kx
          LOGICAL :: nounit
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('STRMV ', info)
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

        SUBROUTINE STRSM(side, uplo, transa, diag, m, n, alpha, a, lda, b,  &
          ldb)
          USE GALAHAD_KINDS
          REAL(r4_) :: alpha
          INTEGER(ip_) :: lda, ldb, m, n
          CHARACTER :: diag, side, transa, uplo
          REAL(r4_) :: a(lda, *), b(ldb, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: MAX
          REAL(r4_) :: temp
          INTEGER(ip_) :: i, info, j, k, nrowa
          LOGICAL :: lside, nounit, upper
          REAL(r4_) :: one, zero
          PARAMETER (one=1.0_r4_, zero=0.0_r4_)
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
            CALL XERBLA('STRSM ', info)
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

        SUBROUTINE STRSV(uplo, trans, diag, n, a, lda, x, incx)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, lda, n
          CHARACTER :: diag, trans, uplo
          REAL(r4_) :: a(lda, *), x(*)
          REAL(r4_) :: zero
          PARAMETER (zero=0.0_r4_)
          REAL(r4_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, kx
          LOGICAL :: nounit
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
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
            CALL XERBLA('STRSV ', info)
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

        SUBROUTINE XERBLA(srname, info)
          USE GALAHAD_KINDS
          CHARACTER (*) :: srname
          INTEGER(ip_) :: info
          INTRINSIC :: LEN_TRIM
          WRITE (*, FMT=9999) srname(1:LEN_TRIM(srname)), info
          STOP
 9999     FORMAT (' ** On entry to ', A, ' parameter number ', I2, ' had ', &
   'an illegal value')
        END SUBROUTINE

        SUBROUTINE ZCOPY(n, zx, incx, zy, incy)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, incy, n
          COMPLEX(c8_) :: zx(*), zy(*)
          INTEGER(ip_) :: i, ix, iy
          IF (n<=0) RETURN
          IF (incx==1 .AND. incy==1) THEN
            DO i = 1, n
              zy(i) = zx(i)
            END DO
          ELSE
            ix = 1
            iy = 1
            IF (incx<0) ix = (-n+1)*incx + 1
            IF (incy<0) iy = (-n+1)*incy + 1
            DO i = 1, n
              zy(iy) = zx(ix)
              ix = ix + incx
              iy = iy + incy
            END DO
          END IF
          RETURN
        END SUBROUTINE

        COMPLEX(c8_) FUNCTION ZDOTC(n, zx, incx, zy, incy)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, incy, n
          COMPLEX(c8_) :: zx(*), zy(*)
          COMPLEX(c8_) :: ztemp
          INTEGER(ip_) :: i, ix, iy
          INTRINSIC :: DCONJG
          ztemp = (0.0_r8_, 0.0_r8_)
          ZDOTC = (0.0_r8_, 0.0_r8_)
          IF (n<=0) RETURN
          IF (incx==1 .AND. incy==1) THEN
            DO i = 1, n
              ztemp = ztemp + DCONJG(zx(i))*zy(i)
            END DO
          ELSE
            ix = 1
            iy = 1
            IF (incx<0) ix = (-n+1)*incx + 1
            IF (incy<0) iy = (-n+1)*incy + 1
            DO i = 1, n
              ztemp = ztemp + DCONJG(zx(ix))*zy(iy)
              ix = ix + incx
              iy = iy + incy
            END DO
          END IF
          ZDOTC = ztemp
          RETURN
        END FUNCTION

        SUBROUTINE ZDSCAL(n, da, zx, incx)
          USE GALAHAD_KINDS
          REAL(r8_) :: da
          INTEGER(ip_) :: incx, n
          COMPLEX(c8_) :: zx(*)
          INTEGER(ip_) :: i, nincx
          INTRINSIC :: DCMPLX
          IF (n<=0 .OR. incx<=0) RETURN
          IF (incx==1) THEN
            DO i = 1, n
              zx(i) = DCMPLX(da, 0.0_r8_)*zx(i)
            END DO
          ELSE
            nincx = n*incx
            DO i = 1, nincx, incx
              zx(i) = DCMPLX(da, 0.0_r8_)*zx(i)
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE ZGEMM(transa, transb, m, n, k, alpha, a, lda, b, ldb,    &
          beta, c, ldc)
          USE GALAHAD_KINDS
          COMPLEX(c8_) :: alpha, beta
          INTEGER(ip_) :: k, lda, ldb, ldc, m, n
          CHARACTER :: transa, transb
          COMPLEX(c8_) :: a(lda, *), b(ldb, *), c(ldc, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: DCONJG, MAX
          COMPLEX(c8_) :: temp
          INTEGER(ip_) :: i, info, j, l, nrowa, nrowb
          LOGICAL :: conja, conjb, nota, notb
          COMPLEX(c8_) :: one
          PARAMETER (one=(1.0_r8_,0.0_r8_))
          COMPLEX(c8_) :: zero
          PARAMETER (zero=(0.0_r8_,0.0_r8_))
          nota = LSAME(transa, 'N')
          notb = LSAME(transb, 'N')
          conja = LSAME(transa, 'C')
          conjb = LSAME(transb, 'C')
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
          IF ((.NOT. nota) .AND. (.NOT. conja) .AND. (.NOT. LSAME(transa,   &
            'T'))) THEN
            info = 1
          ELSE IF ((.NOT. notb) .AND. (.NOT. conjb) .AND. (.NOT. LSAME(     &
            transb,'T'))) THEN
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
            CALL XERBLA('ZGEMM ', info)
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
            ELSE IF (conja) THEN
              DO j = 1, n
                DO i = 1, m
                  temp = zero
                  DO l = 1, k
                    temp = temp + DCONJG(a(l,i))*b(l, j)
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
          ELSE IF (nota) THEN
            IF (conjb) THEN
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
                  temp = alpha*DCONJG(b(j,l))
                  DO i = 1, m
                    c(i, j) = c(i, j) + temp*a(i, l)
                  END DO
                END DO
              END DO
            ELSE
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
            END IF
          ELSE IF (conja) THEN
            IF (conjb) THEN
              DO j = 1, n
                DO i = 1, m
                  temp = zero
                  DO l = 1, k
                    temp = temp + DCONJG(a(l,i))*DCONJG(b(j,l))
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
                DO i = 1, m
                  temp = zero
                  DO l = 1, k
                    temp = temp + DCONJG(a(l,i))*b(j, l)
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
            IF (conjb) THEN
              DO j = 1, n
                DO i = 1, m
                  temp = zero
                  DO l = 1, k
                    temp = temp + a(l, i)*DCONJG(b(j,l))
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

        SUBROUTINE ZGEMV(trans, m, n, alpha, a, lda, x, incx, beta, y,      &
          incy)
          USE GALAHAD_KINDS
          COMPLEX(c8_) :: alpha, beta
          INTEGER(ip_) :: incx, incy, lda, m, n
          CHARACTER :: trans
          COMPLEX(c8_) :: a(lda, *), x(*), y(*)
          COMPLEX(c8_) :: one
          PARAMETER (one=(1.0_r8_,0.0_r8_))
          COMPLEX(c8_) :: zero
          PARAMETER (zero=(0.0_r8_,0.0_r8_))
          COMPLEX(c8_) :: temp
          INTEGER(ip_) :: i, info, ix, iy, j, jx, jy, kx, ky, lenx, leny
          LOGICAL :: noconj
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: DCONJG, MAX
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
            CALL XERBLA('ZGEMV ', info)
            RETURN
          END IF
          IF ((m==0) .OR. (n==0) .OR. ((alpha==zero) .AND. (beta== one)))   &
            RETURN
          noconj = LSAME(trans, 'T')
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
                IF (noconj) THEN
                  DO i = 1, m
                    temp = temp + a(i, j)*x(i)
                  END DO
                ELSE
                  DO i = 1, m
                    temp = temp + DCONJG(a(i,j))*x(i)
                  END DO
                END IF
                y(jy) = y(jy) + alpha*temp
                jy = jy + incy
              END DO
            ELSE
              DO j = 1, n
                temp = zero
                ix = kx
                IF (noconj) THEN
                  DO i = 1, m
                    temp = temp + a(i, j)*x(ix)
                    ix = ix + incx
                  END DO
                ELSE
                  DO i = 1, m
                    temp = temp + DCONJG(a(i,j))*x(ix)
                    ix = ix + incx
                  END DO
                END IF
                y(jy) = y(jy) + alpha*temp
                jy = jy + incy
              END DO
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE ZGERC(m, n, alpha, x, incx, y, incy, a, lda)
          USE GALAHAD_KINDS
          COMPLEX(c8_) :: alpha
          INTEGER(ip_) :: incx, incy, lda, m, n
          COMPLEX(c8_) :: a(lda, *), x(*), y(*)
          COMPLEX(c8_) :: zero
          PARAMETER (zero=(0.0_r8_,0.0_r8_))
          COMPLEX(c8_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jy, kx
          EXTERNAL :: XERBLA
          INTRINSIC :: DCONJG, MAX
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
            CALL XERBLA('ZGERC ', info)
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
                temp = alpha*DCONJG(y(jy))
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
                temp = alpha*DCONJG(y(jy))
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

        SUBROUTINE ZGERU(m, n, alpha, x, incx, y, incy, a, lda)
          USE GALAHAD_KINDS
          COMPLEX(c8_) :: alpha
          INTEGER(ip_) :: incx, incy, lda, m, n
          COMPLEX(c8_) :: a(lda, *), x(*), y(*)
          COMPLEX(c8_) :: zero
          PARAMETER (zero=(0.0_r8_,0.0_r8_))
          COMPLEX(c8_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jy, kx
          EXTERNAL :: XERBLA
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
            CALL XERBLA('ZGERU ', info)
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

        SUBROUTINE ZHER(uplo, n, alpha, x, incx, a, lda)
          USE GALAHAD_KINDS
          REAL(r8_) :: alpha
          INTEGER(ip_) :: incx, lda, n
          CHARACTER :: uplo
          COMPLEX(c8_) :: a(lda, *), x(*)
          COMPLEX(c8_) :: zero
          PARAMETER (zero=(0.0_r8_,0.0_r8_))
          COMPLEX(c8_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, kx
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: DBLE, DCONJG, MAX
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
            CALL XERBLA('ZHER  ', info)
            RETURN
          END IF
          IF ((n==0) .OR. (alpha==DBLE(zero))) RETURN
          IF (incx<=0) THEN
            kx = 1 - (n-1)*incx
          ELSE IF (incx/=1) THEN
            kx = 1
          END IF
          IF (LSAME(uplo,'U')) THEN
            IF (incx==1) THEN
              DO j = 1, n
                IF (x(j)/=zero) THEN
                  temp = alpha*DCONJG(x(j))
                  DO i = 1, j - 1
                    a(i, j) = a(i, j) + x(i)*temp
                  END DO
                  a(j, j) = DBLE(a(j,j)) + DBLE(x(j)*temp)
                ELSE
                  a(j, j) = DBLE(a(j,j))
                END IF
              END DO
            ELSE
              jx = kx
              DO j = 1, n
                IF (x(jx)/=zero) THEN
                  temp = alpha*DCONJG(x(jx))
                  ix = kx
                  DO i = 1, j - 1
                    a(i, j) = a(i, j) + x(ix)*temp
                    ix = ix + incx
                  END DO
                  a(j, j) = DBLE(a(j,j)) + DBLE(x(jx)*temp)
                ELSE
                  a(j, j) = DBLE(a(j,j))
                END IF
                jx = jx + incx
              END DO
            END IF
          ELSE
            IF (incx==1) THEN
              DO j = 1, n
                IF (x(j)/=zero) THEN
                  temp = alpha*DCONJG(x(j))
                  a(j, j) = DBLE(a(j,j)) + DBLE(temp*x(j))
                  DO i = j + 1, n
                    a(i, j) = a(i, j) + x(i)*temp
                  END DO
                ELSE
                  a(j, j) = DBLE(a(j,j))
                END IF
              END DO
            ELSE
              jx = kx
              DO j = 1, n
                IF (x(jx)/=zero) THEN
                  temp = alpha*DCONJG(x(jx))
                  a(j, j) = DBLE(a(j,j)) + DBLE(temp*x(jx))
                  ix = jx
                  DO i = j + 1, n
                    ix = ix + incx
                    a(i, j) = a(i, j) + x(ix)*temp
                  END DO
                ELSE
                  a(j, j) = DBLE(a(j,j))
                END IF
                jx = jx + incx
              END DO
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE ZSCAL(n, za, zx, incx)
          USE GALAHAD_KINDS
          COMPLEX(c8_) :: za
          INTEGER(ip_) :: incx, n
          COMPLEX(c8_) :: zx(*)
          INTEGER(ip_) :: i, nincx
          IF (n<=0 .OR. incx<=0) RETURN
          IF (incx==1) THEN
            DO i = 1, n
              zx(i) = za*zx(i)
            END DO
          ELSE
            nincx = n*incx
            DO i = 1, nincx, incx
              zx(i) = za*zx(i)
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE ZSWAP(n, zx, incx, zy, incy)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, incy, n
          COMPLEX(c8_) :: zx(*), zy(*)
          COMPLEX(c8_) :: ztemp
          INTEGER(ip_) :: i, ix, iy
          IF (n<=0) RETURN
          IF (incx==1 .AND. incy==1) THEN
            DO i = 1, n
              ztemp = zx(i)
              zx(i) = zy(i)
              zy(i) = ztemp
            END DO
          ELSE
            ix = 1
            iy = 1
            IF (incx<0) ix = (-n+1)*incx + 1
            IF (incy<0) iy = (-n+1)*incy + 1
            DO i = 1, n
              ztemp = zx(ix)
              zx(ix) = zy(iy)
              zy(iy) = ztemp
              ix = ix + incx
              iy = iy + incy
            END DO
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE ZTRMM(side, uplo, transa, diag, m, n, alpha, a, lda, b,  &
          ldb)
          USE GALAHAD_KINDS
          COMPLEX(c8_) :: alpha
          INTEGER(ip_) :: lda, ldb, m, n
          CHARACTER :: diag, side, transa, uplo
          COMPLEX(c8_) :: a(lda, *), b(ldb, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: DCONJG, MAX
          COMPLEX(c8_) :: temp
          INTEGER(ip_) :: i, info, j, k, nrowa
          LOGICAL :: lside, noconj, nounit, upper
          COMPLEX(c8_) :: one
          PARAMETER (one=(1.0_r8_,0.0_r8_))
          COMPLEX(c8_) :: zero
          PARAMETER (zero=(0.0_r8_,0.0_r8_))
          lside = LSAME(side, 'L')
          IF (lside) THEN
            nrowa = m
          ELSE
            nrowa = n
          END IF
          noconj = LSAME(transa, 'T')
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
            CALL XERBLA('ZTRMM ', info)
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
                    IF (noconj) THEN
                      IF (nounit) temp = temp*a(i, i)
                      DO k = 1, i - 1
                        temp = temp + a(k, i)*b(k, j)
                      END DO
                    ELSE
                      IF (nounit) temp = temp*DCONJG(a(i,i))
                      DO k = 1, i - 1
                        temp = temp + DCONJG(a(k,i))*b(k, j)
                      END DO
                    END IF
                    b(i, j) = alpha*temp
                  END DO
                END DO
              ELSE
                DO j = 1, n
                  DO i = 1, m
                    temp = b(i, j)
                    IF (noconj) THEN
                      IF (nounit) temp = temp*a(i, i)
                      DO k = i + 1, m
                        temp = temp + a(k, i)*b(k, j)
                      END DO
                    ELSE
                      IF (nounit) temp = temp*DCONJG(a(i,i))
                      DO k = i + 1, m
                        temp = temp + DCONJG(a(k,i))*b(k, j)
                      END DO
                    END IF
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
                      IF (noconj) THEN
                        temp = alpha*a(j, k)
                      ELSE
                        temp = alpha*DCONJG(a(j,k))
                      END IF
                      DO i = 1, m
                        b(i, j) = b(i, j) + temp*b(i, k)
                      END DO
                    END IF
                  END DO
                  temp = alpha
                  IF (nounit) THEN
                    IF (noconj) THEN
                      temp = temp*a(k, k)
                    ELSE
                      temp = temp*DCONJG(a(k,k))
                    END IF
                  END IF
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
                      IF (noconj) THEN
                        temp = alpha*a(j, k)
                      ELSE
                        temp = alpha*DCONJG(a(j,k))
                      END IF
                      DO i = 1, m
                        b(i, j) = b(i, j) + temp*b(i, k)
                      END DO
                    END IF
                  END DO
                  temp = alpha
                  IF (nounit) THEN
                    IF (noconj) THEN
                      temp = temp*a(k, k)
                    ELSE
                      temp = temp*DCONJG(a(k,k))
                    END IF
                  END IF
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

        SUBROUTINE ZTRMV(uplo, trans, diag, n, a, lda, x, incx)
          USE GALAHAD_KINDS
          INTEGER(ip_) :: incx, lda, n
          CHARACTER :: diag, trans, uplo
          COMPLEX(c8_) :: a(lda, *), x(*)
          COMPLEX(c8_) :: zero
          PARAMETER (zero=(0.0_r8_,0.0_r8_))
          COMPLEX(c8_) :: temp
          INTEGER(ip_) :: i, info, ix, j, jx, kx
          LOGICAL :: noconj, nounit
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: DCONJG, MAX
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
            CALL XERBLA('ZTRMV ', info)
            RETURN
          END IF
          IF (n==0) RETURN
          noconj = LSAME(trans, 'T')
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
                  IF (noconj) THEN
                    IF (nounit) temp = temp*a(j, j)
                    DO i = j - 1, 1_ip_, -1_ip_
                      temp = temp + a(i, j)*x(i)
                    END DO
                  ELSE
                    IF (nounit) temp = temp*DCONJG(a(j,j))
                    DO i = j - 1, 1_ip_, -1_ip_
                      temp = temp + DCONJG(a(i,j))*x(i)
                    END DO
                  END IF
                  x(j) = temp
                END DO
              ELSE
                jx = kx + (n-1)*incx
                DO j = n, 1_ip_, -1_ip_
                  temp = x(jx)
                  ix = jx
                  IF (noconj) THEN
                    IF (nounit) temp = temp*a(j, j)
                    DO i = j - 1, 1_ip_, -1_ip_
                      ix = ix - incx
                      temp = temp + a(i, j)*x(ix)
                    END DO
                  ELSE
                    IF (nounit) temp = temp*DCONJG(a(j,j))
                    DO i = j - 1, 1_ip_, -1_ip_
                      ix = ix - incx
                      temp = temp + DCONJG(a(i,j))*x(ix)
                    END DO
                  END IF
                  x(jx) = temp
                  jx = jx - incx
                END DO
              END IF
            ELSE
              IF (incx==1) THEN
                DO j = 1, n
                  temp = x(j)
                  IF (noconj) THEN
                    IF (nounit) temp = temp*a(j, j)
                    DO i = j + 1, n
                      temp = temp + a(i, j)*x(i)
                    END DO
                  ELSE
                    IF (nounit) temp = temp*DCONJG(a(j,j))
                    DO i = j + 1, n
                      temp = temp + DCONJG(a(i,j))*x(i)
                    END DO
                  END IF
                  x(j) = temp
                END DO
              ELSE
                jx = kx
                DO j = 1, n
                  temp = x(jx)
                  ix = jx
                  IF (noconj) THEN
                    IF (nounit) temp = temp*a(j, j)
                    DO i = j + 1, n
                      ix = ix + incx
                      temp = temp + a(i, j)*x(ix)
                    END DO
                  ELSE
                    IF (nounit) temp = temp*DCONJG(a(j,j))
                    DO i = j + 1, n
                      ix = ix + incx
                      temp = temp + DCONJG(a(i,j))*x(ix)
                    END DO
                  END IF
                  x(jx) = temp
                  jx = jx + incx
                END DO
              END IF
            END IF
          END IF
          RETURN
        END SUBROUTINE

        SUBROUTINE ZTRSM(side, uplo, transa, diag, m, n, alpha, a, lda, b,  &
          ldb)
          USE GALAHAD_KINDS
          COMPLEX(c8_) :: alpha
          INTEGER(ip_) :: lda, ldb, m, n
          CHARACTER :: diag, side, transa, uplo
          COMPLEX(c8_) :: a(lda, *), b(ldb, *)
          LOGICAL :: LSAME
          EXTERNAL :: LSAME
          EXTERNAL :: XERBLA
          INTRINSIC :: DCONJG, MAX
          COMPLEX(c8_) :: temp
          INTEGER(ip_) :: i, info, j, k, nrowa
          LOGICAL :: lside, noconj, nounit, upper
          COMPLEX(c8_) :: one
          PARAMETER (one=(1.0_r8_,0.0_r8_))
          COMPLEX(c8_) :: zero
          PARAMETER (zero=(0.0_r8_,0.0_r8_))
          lside = LSAME(side, 'L')
          IF (lside) THEN
            nrowa = m
          ELSE
            nrowa = n
          END IF
          noconj = LSAME(transa, 'T')
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
            CALL XERBLA('ZTRSM ', info)
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
                    IF (noconj) THEN
                      DO k = 1, i - 1
                        temp = temp - a(k, i)*b(k, j)
                      END DO
                      IF (nounit) temp = temp/a(i, i)
                    ELSE
                      DO k = 1, i - 1
                        temp = temp - DCONJG(a(k,i))*b(k, j)
                      END DO
                      IF (nounit) temp = temp/DCONJG(a(i,i))
                    END IF
                    b(i, j) = temp
                  END DO
                END DO
              ELSE
                DO j = 1, n
                  DO i = m, 1_ip_, -1_ip_
                    temp = alpha*b(i, j)
                    IF (noconj) THEN
                      DO k = i + 1, m
                        temp = temp - a(k, i)*b(k, j)
                      END DO
                      IF (nounit) temp = temp/a(i, i)
                    ELSE
                      DO k = i + 1, m
                        temp = temp - DCONJG(a(k,i))*b(k, j)
                      END DO
                      IF (nounit) temp = temp/DCONJG(a(i,i))
                    END IF
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
                    IF (noconj) THEN
                      temp = one/a(k, k)
                    ELSE
                      temp = one/DCONJG(a(k,k))
                    END IF
                    DO i = 1, m
                      b(i, k) = temp*b(i, k)
                    END DO
                  END IF
                  DO j = 1, k - 1
                    IF (a(j,k)/=zero) THEN
                      IF (noconj) THEN
                        temp = a(j, k)
                      ELSE
                        temp = DCONJG(a(j,k))
                      END IF
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
                    IF (noconj) THEN
                      temp = one/a(k, k)
                    ELSE
                      temp = one/DCONJG(a(k,k))
                    END IF
                    DO i = 1, m
                      b(i, k) = temp*b(i, k)
                    END DO
                  END IF
                  DO j = k + 1, n
                    IF (a(j,k)/=zero) THEN
                      IF (noconj) THEN
                        temp = a(j, k)
                      ELSE
                        temp = DCONJG(a(j,k))
                      END IF
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
