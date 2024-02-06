! THIS VERSION: GALAHAD 4.3 - 2023-02-06 AT 11:30 GMT.

! Define BLAS API in Fortran module

#ifdef INTEGER_64
#define spral_ssids_blas_iface spral_ssids_blas_iface_64
#define spral_kinds spral_kinds_64
#endif

module spral_ssids_blas_iface
  implicit none

  private
  public :: sgemv, strsv
  public :: dgemv, dtrsv
  public :: sgemm, ssyrk, strsm
  public :: dgemm, dsyrk, dtrsm

  ! Level 2 BLAS
  interface
    subroutine sgemv( trans, m, n, alpha, a, lda, x, incx, beta, y, incy )
      use spral_kinds, only: ip_, sp_
      implicit none
      character, intent(in) :: trans
      integer(ip_), intent(in) :: m, n, lda, incx, incy
      real(sp_), intent(in) :: alpha, beta
      real(sp_), intent(in   ), dimension(lda, n) :: a
      real(sp_), intent(in   ), dimension(*) :: x
      real(sp_), intent(inout), dimension(*) :: y
    end subroutine sgemv
    subroutine strsv( uplo, trans, diag, n, a, lda, x, incx )
      use spral_kinds, only: ip_, sp_
      implicit none
      character, intent(in) :: uplo, trans, diag
      integer(ip_), intent(in) :: n, lda, incx
      real(sp_), intent(in   ), dimension(lda, n) :: a
      real(sp_), intent(inout), dimension(*) :: x
    end subroutine strsv
  end interface

  interface
    subroutine dgemv( trans, m, n, alpha, a, lda, x, incx, beta, y, incy )
      use spral_kinds, only: ip_, dp_
      implicit none
      character, intent(in) :: trans
      integer(ip_), intent(in) :: m, n, lda, incx, incy
      real(dp_), intent(in) :: alpha, beta
      real(dp_), intent(in   ), dimension(lda, n) :: a
      real(dp_), intent(in   ), dimension(*) :: x
      real(dp_), intent(inout), dimension(*) :: y
    end subroutine dgemv
    subroutine dtrsv( uplo, trans, diag, n, a, lda, x, incx )
      use spral_kinds, only: ip_, dp_
      implicit none
      character, intent(in) :: uplo, trans, diag
      integer(ip_), intent(in) :: n, lda, incx
      real(dp_), intent(in   ), dimension(lda, n) :: a
      real(dp_), intent(inout), dimension(*) :: x
    end subroutine dtrsv
  end interface

  ! Level 3 BLAS
  interface
    subroutine sgemm( ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc )
      use spral_kinds, only : ip_, sp_
      implicit none
      character, intent(in) :: ta, tb
      integer(ip_), intent(in) :: m, n, k
      integer(ip_), intent(in) :: lda, ldb, ldc
      real(sp_), intent(in) :: alpha, beta
      real(sp_), intent(in   ), dimension(lda, *) :: a
      real(sp_), intent(in   ), dimension(ldb, *) :: b
      real(sp_), intent(inout), dimension(ldc, *) :: c
    end subroutine sgemm
    subroutine ssyrk( uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
      use spral_kinds, only : ip_, sp_
      implicit none
      character, intent(in) :: uplo, trans
      integer(ip_), intent(in) :: n, k, lda, ldc
      real(sp_), intent(in) :: alpha, beta
      real(sp_), intent(in   ), dimension(lda, *) :: a
      real(sp_), intent(inout), dimension(ldc, n) :: c
    end subroutine ssyrk
    subroutine strsm( side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb )
      use spral_kinds, only : ip_, sp_
      implicit none
      character, intent(in) :: side, uplo, trans, diag
      integer(ip_), intent(in) :: m, n, lda, ldb
      real(sp_), intent(in   ) :: alpha
      real(sp_), intent(in   ) :: a(lda, *)
      real(sp_), intent(inout) :: b(ldb, n)
    end subroutine strsm
  end interface

  interface
    subroutine dgemm( ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc )
      use spral_kinds, only : ip_, dp_
      implicit none
      character, intent(in) :: ta, tb
      integer(ip_), intent(in) :: m, n, k
      integer(ip_), intent(in) :: lda, ldb, ldc
      real(dp_), intent(in) :: alpha, beta
      real(dp_), intent(in   ), dimension(lda, *) :: a
      real(dp_), intent(in   ), dimension(ldb, *) :: b
      real(dp_), intent(inout), dimension(ldc, *) :: c
    end subroutine dgemm
    subroutine dsyrk( uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
      use spral_kinds, only : ip_, dp_
      implicit none
      character, intent(in) :: uplo, trans
      integer(ip_), intent(in) :: n, k, lda, ldc
      real(dp_), intent(in) :: alpha, beta
      real(dp_), intent(in   ), dimension(lda, *) :: a
      real(dp_), intent(inout), dimension(ldc, n) :: c
    end subroutine dsyrk
    subroutine dtrsm( side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb )
      use spral_kinds, only : ip_, dp_
      implicit none
      character, intent(in) :: side, uplo, trans, diag
      integer(ip_), intent(in) :: m, n, lda, ldb
      real(dp_), intent(in   ) :: alpha
      real(dp_), intent(in   ) :: a(lda, *)
      real(dp_), intent(inout) :: b(ldb, n)
    end subroutine dtrsm
  end interface

end module spral_ssids_blas_iface
