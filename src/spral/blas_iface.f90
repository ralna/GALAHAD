! Define BLAS API in Fortran module
module spral_blas_iface
  implicit none

  private
  public :: daxpy, dcopy, ddot, dnrm2, dscal
  public :: zaxpy, zcopy, zdotc, dznrm2, zscal
  public :: dgemv, dtrsv
  public :: dgemm, dsyrk, dtrsm
  public :: zgemm, ztrsm

  ! Level 1 BLAS
  interface
    subroutine daxpy( n, a, x, incx, y, incy )
      implicit none
      integer, intent(in) :: n, incx, incy
      double precision, intent(in) :: a
      double precision, intent(in   ), dimension(*) :: x
      double precision, intent(inout), dimension(*) :: y
    end subroutine daxpy
    subroutine dcopy( n, x, incx, y, incy )
      implicit none
      integer, intent(in) :: n, incx, incy
      double precision, intent(in ), dimension(*) :: x
      double precision, intent(out), dimension(*) :: y
    end subroutine dcopy
    double precision function ddot( n, x, incx, y, incy )
      implicit none
      integer, intent(in) :: n, incx, incy
      double precision, intent(in), dimension(*) :: x
      double precision, intent(in), dimension(*) :: y
    end function ddot
    double precision function dnrm2( n, x, incx )
      implicit none
      integer, intent(in) :: n, incx
      double precision, intent(in), dimension(*) :: x
    end function dnrm2
    subroutine dscal( n, a, x, incx )
      implicit none
      integer, intent(in) :: n, incx
      double precision, intent(in) :: a
      double precision, intent(in), dimension(*) :: x
    end subroutine dscal
    subroutine zcopy( n, x, incx, y, incy )
      implicit none
      integer, parameter :: PRECISION = kind(1.0D0)
      integer, intent(in) :: n, incx, incy
      complex(PRECISION), intent(in ), dimension(*) :: x
      complex(PRECISION), intent(out), dimension(*) :: y
    end subroutine zcopy
    double precision function dznrm2( n, x, incx )
      implicit none
      integer, parameter :: PRECISION = kind(1.0D0)
      integer, intent(in) :: n, incx
      complex(PRECISION), intent(in), dimension(*) :: x
    end function dznrm2
    function zdotc( n, x, incx, y, incy )
      implicit none
      integer, parameter :: PRECISION = kind(1.0D0)
      integer, intent(in) :: n, incx, incy
      complex(PRECISION), intent(in), dimension(*) :: x
      complex(PRECISION), intent(in), dimension(*) :: y
      complex(PRECISION) :: zdotc
    end function zdotc
    subroutine zscal( n, a, x, incx )
      implicit none
      integer, parameter :: PRECISION = kind(1.0D0)
      integer, intent(in) :: n, incx
      complex(PRECISION), intent(in) :: a
      complex(PRECISION), intent(in), dimension(*) :: x
    end subroutine zscal
    subroutine zaxpy( n, a, x, incx, y, incy )
      implicit none
      integer, parameter :: PRECISION = kind(1.0D0)
      integer, intent(in) :: n, incx, incy
      complex(PRECISION), intent(in) :: a
      complex(PRECISION), intent(in   ), dimension(*) :: x
      complex(PRECISION), intent(inout), dimension(*) :: y
    end subroutine zaxpy
  end interface

  ! Level 2 BLAS
  interface
    subroutine dgemv( trans, m, n, alpha, a, lda, x, incx, beta, y, incy )
      implicit none
      character, intent(in) :: trans
      integer, intent(in) :: m, n, lda, incx, incy
      double precision, intent(in) :: alpha, beta
      double precision, intent(in   ), dimension(lda, n) :: a
      double precision, intent(in   ), dimension(*) :: x
      double precision, intent(inout), dimension(*) :: y
    end subroutine dgemv
    subroutine dtrsv( uplo, trans, diag, n, a, lda, x, incx )
      implicit none
      character, intent(in) :: uplo, trans, diag
      integer, intent(in) :: n, lda, incx
      double precision, intent(in   ), dimension(lda, n) :: a
      double precision, intent(inout), dimension(*) :: x
    end subroutine dtrsv
  end interface

  ! Level 3 BLAS
  interface
    subroutine dgemm( ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc )
      implicit none
      character, intent(in) :: ta, tb
      integer, intent(in) :: m, n, k
      integer, intent(in) :: lda, ldb, ldc
      double precision, intent(in) :: alpha, beta
      double precision, intent(in   ), dimension(lda, *) :: a
      double precision, intent(in   ), dimension(ldb, *) :: b
      double precision, intent(inout), dimension(ldc, *) :: c
    end subroutine dgemm
    subroutine dsyrk( uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
      implicit none
      character, intent(in) :: uplo, trans
      integer, intent(in) :: n, k, lda, ldc
      double precision, intent(in) :: alpha, beta
      double precision, intent(in   ), dimension(lda, *) :: a
      double precision, intent(inout), dimension(ldc, n) :: c
    end subroutine dsyrk
    subroutine dtrsm( side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb )
      implicit none
      character, intent(in) :: side, uplo, trans, diag
      integer, intent(in) :: m, n, lda, ldb
      double precision, intent(in   ) :: alpha
      double precision, intent(in   ) :: a(lda, *)
      double precision, intent(inout) :: b(ldb, n)
    end subroutine dtrsm
    subroutine zgemm( ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc )
      implicit none
      integer, parameter :: PRECISION = kind(1.0D0)
      character, intent(in) :: ta, tb
      integer, intent(in) :: m, n, k
      integer, intent(in) :: lda, ldb, ldc
      complex(PRECISION), intent(in) :: alpha, beta
      complex(PRECISION), intent(in   ), dimension(lda, *) :: a
      complex(PRECISION), intent(in   ), dimension(ldb, *) :: b
      complex(PRECISION), intent(inout), dimension(ldc, *) :: c
    end subroutine zgemm
    subroutine ztrsm( side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb )
      implicit none
      integer, parameter :: PRECISION = kind(1.0D0)
      character, intent(in) :: side, uplo, trans, diag
      integer, intent(in) :: m, n, lda, ldb
      complex(PRECISION), intent(in   ) :: alpha
      complex(PRECISION), intent(in   ) :: a(lda, *)
      complex(PRECISION), intent(inout) :: b(ldb, n)
    end subroutine ztrsm
  end interface

end module spral_blas_iface
