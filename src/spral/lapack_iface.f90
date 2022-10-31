! Definition of LAPACK API in module
module spral_lapack_iface
  implicit none

  private
  public :: dpotrf, dlacpy, dsytrf
  public :: zpotrf, zlacpy

  interface
    subroutine dpotrf( uplo, n, a, lda, info )
      implicit none
      character, intent(in) :: uplo
      integer, intent(in) :: n, lda
      double precision, intent(inout) :: a(lda, n)
      integer, intent(out) :: info
    end subroutine dpotrf
    subroutine dlacpy( uplo, m, n, a, lda, b, ldb )
      implicit none
      character, intent(in) :: uplo
      integer, intent(in) :: m, n, lda, ldb
      double precision, intent(in ) :: a(lda, n)
      double precision, intent(out) :: b(ldb, n)
    end subroutine dlacpy
    subroutine dsytrf( uplo, n, a, lda, ipiv, work, lwork, info )
      implicit none
      character, intent(in) :: uplo
      integer, intent(in) :: n, lda, lwork
      integer, intent(out), dimension(n) :: ipiv
      integer, intent(out) :: info
      double precision, intent(inout), dimension(lda, *) :: a
      double precision, intent(out  ), dimension(*) :: work
    end subroutine dsytrf
  end interface

  interface
    subroutine zpotrf( uplo, n, a, lda, info )
      implicit none
      integer, parameter :: PRECISION = kind(1.0D0)
      character, intent(in) :: uplo
      integer, intent(in) :: n, lda
      complex(PRECISION), intent(inout) :: a(lda, n)
      integer, intent(out) :: info
    end subroutine zpotrf
    subroutine zlacpy( uplo, m, n, a, lda, b, ldb )
      implicit none
      integer, parameter :: PRECISION = kind(1.0D0)
      character, intent(in) :: uplo
      integer, intent(in) :: m, n, lda, ldb
      complex(PRECISION), intent(in ) :: a(lda, n)
      complex(PRECISION), intent(out) :: b(ldb, n)
    end subroutine zlacpy
  end interface

end module spral_lapack_iface
