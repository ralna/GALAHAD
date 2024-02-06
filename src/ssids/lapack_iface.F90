! THIS VERSION: GALAHAD 4.3 - 2024-02-06 AT 11:30 GMT.

#ifdef INTEGER_64
#define spral_ssids_lapack_iface spral_ssids_lapack_iface_64
#define spral_kinds spral_kinds_64
#endif

! Definition of LAPACK API in module

module spral_ssids_lapack_iface
  implicit none

  private
  public :: spotrf, ssytrf
  public :: dpotrf, dsytrf

  interface
    subroutine spotrf( uplo, n, a, lda, info )
      use spral_kinds, only: ip_, sp_
      implicit none
      character, intent(in) :: uplo
      integer(ip_), intent(in) :: n, lda
      real(sp_), intent(inout) :: a(lda, n)
      integer(ip_), intent(out) :: info
    end subroutine spotrf
    subroutine ssytrf( uplo, n, a, lda, ipiv, work, lwork, info )
      use spral_kinds, only: ip_, sp_
      implicit none
      character, intent(in) :: uplo
      integer(ip_), intent(in) :: n, lda, lwork
      integer(ip_), intent(out), dimension(n) :: ipiv
      integer(ip_), intent(out) :: info
      real(sp_), intent(inout), dimension(lda, *) :: a
      real(sp_), intent(out  ), dimension(*) :: work
    end subroutine ssytrf
  end interface

  interface
    subroutine dpotrf( uplo, n, a, lda, info )
      use spral_kinds, only: ip_, dp_
      implicit none
      character, intent(in) :: uplo
      integer(ip_), intent(in) :: n, lda
      real(dp_), intent(inout) :: a(lda, n)
      integer(ip_), intent(out) :: info
    end subroutine dpotrf
    subroutine dsytrf( uplo, n, a, lda, ipiv, work, lwork, info )
      use spral_kinds, only: ip_, dp_
      implicit none
      character, intent(in) :: uplo
      integer(ip_), intent(in) :: n, lda, lwork
      integer(ip_), intent(out), dimension(n) :: ipiv
      integer(ip_), intent(out) :: info
      real(dp_), intent(inout), dimension(lda, *) :: a
      real(dp_), intent(out  ), dimension(*) :: work
    end subroutine dsytrf
  end interface

end module spral_ssids_lapack_iface
