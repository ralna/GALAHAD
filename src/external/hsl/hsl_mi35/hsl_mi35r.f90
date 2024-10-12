! THIS VERSION: GALAHAD 5.1 - 2024-10-11 AT 14:30 GMT.

#include "hsl_subset.h"

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M I 3 5   M O D U L E  -*-*-*-

  module hsl_mi35_real
    use hsl_kinds_real, only: ip_, long_, lp_, rp_
#ifdef INTEGER_64
     USE GALAHAD_SYMBOLS_64, ONLY: GALAHAD_unavailable_option
#else
     USE GALAHAD_SYMBOLS, ONLY: GALAHAD_unavailable_option
#endif

    implicit none

    private
    public :: mi35_keep, mi35_control, mi35_info
    public :: mi35_factorize, mi35_finalise, mi35_precondition, mi35_solve
    public :: mi35_check_matrix, mi35_factorizeC, mi35_formC
    LOGICAL, PUBLIC, PARAMETER :: mi35_available = .FALSE.

    real(rp_), parameter :: zero = 0.0_rp_
    real(rp_), parameter :: one = 1.0_rp_
    real(rp_), parameter :: sfact = 2.0_rp_
    real(rp_), parameter :: sfact2 = 4.0_rp_
    real(rp_), parameter :: alpham = 0.001_rp_

    type mi35_keep
      integer(long_), allocatable ::  fact_ptr(:)
      integer(ip_),  allocatable ::  fact_row(:)
      real(rp_), allocatable ::  fact_val(:)
      real(rp_), allocatable :: scale(:)
      integer(ip_),  allocatable :: invp(:)
      integer(ip_),  allocatable :: perm(:)
      real(rp_), allocatable :: w(:)
    end type mi35_keep

    type mi35_control
      real(rp_) :: alpha = zero
      integer(ip_) :: iorder = 6
      integer(ip_) :: iscale = 1
      integer(ip_) :: limit_rowA = -1
      integer(ip_) :: limit_colC = -1
      integer(ip_) :: limit_C = -1
      real(rp_) :: lowalpha = alpham
      integer(ip_) :: maxshift = 3
      logical(lp_) :: rrt = .false.
      real(rp_) :: shift_factor = sfact
      real(rp_) :: shift_factor2 = sfact2
      real(rp_) :: small = 10.0_rp_**(-20)
      real(rp_) :: tau1 = 0.001_rp_
      real(rp_) :: tau2 = 0.0001_rp_
      integer(ip_) :: unit_error = 6
      integer(ip_) :: unit_warning = 6
    end type mi35_control

    type mi35_info
      real(rp_) :: avlenC = zero
      integer(ip_) :: band_after = 0
      integer(ip_) :: band_before = 0
      integer(ip_) :: dup = 0
      integer(ip_) :: flag = 0
      integer(ip_) :: flag61 = 0
      integer(ip_) :: flag64 = 0
      integer(ip_) :: flag68 = 0
      integer(ip_) :: flag77 = 0
      integer(ip_) :: maxlen = 0
      integer(ip_) :: maxlenC = 0
      integer(ip_) :: nrestart = 0
      integer(ip_) :: nshift = 0
      integer(ip_) :: nnz_C = 0
      integer(ip_) :: nzero = 0
      integer(ip_) :: nzero_weight = 0
      integer(ip_) :: oor = 0
      real(rp_) :: profile_before = 0
      real(rp_) :: profile_after = 0
      integer(long_) :: size_r = 0_long_
      integer(ip_) :: stat = 0
      real(rp_) :: alpha = zero
    end type mi35_info

  contains

    subroutine mi35_check_matrix( m, n, ptr, row, val, control, info, weight, b)
    integer(ip_),  intent(inout) :: m
    integer(ip_),  intent(inout) :: n
    integer(ip_),  intent(inout) ::  ptr(n+1)
    integer(ip_),  intent(inout) ::  row(:)
    real(rp_), intent(inout) ::  val(:)
    type(mi35_info), intent(out) :: info
    type(mi35_control), intent(in) :: control
    real(rp_), intent(inout), optional :: weight(m)
    real(rp_), intent(inout), optional :: b(m)
    IF ( control%unit_error >= 0 ) WRITE( control%unit_error,                  &
         "( ' We regret that the solution options that you have ', /,          &
   &     ' chosen are not all freely available with GALAHAD.', /,              &
   &     ' If you have HSL (formerly the Harwell Subroutine', /,               &
   &     ' Library), this option may be enabled by replacing the dummy ', /,   &
   &     ' subroutine MI35_check_matrix with its HSL namesake ', /,            &
   &     ' and dependencies. See ', /,                                         &
   &     '   $GALAHAD/src/makedefs/packages for details.' )" )
    info%flag = GALAHAD_unavailable_option
    end subroutine mi35_check_matrix

!****************************************************************************

    subroutine mi35_factorize( m, n, ptr, row, val, lsize, rsize, keep,        &
                               control, info, weight, scale, perm )
    integer(ip_),  intent(in) :: m
    integer(ip_),  intent(in) :: n
    integer(ip_),  intent(in) ::  ptr(n+1)
    integer(ip_),  intent(in) ::  row(:)
    real(rp_), intent(in) ::  val(:)
    integer(ip_),  intent(in) :: lsize
    integer(ip_),  intent(in) :: rsize
    type(mi35_keep), intent(out) :: keep
    type(mi35_control), intent(in) :: control
    type(mi35_info), intent(out) :: info
    real(rp_), intent(in), optional :: weight(m)
    real(rp_), intent(in), optional :: scale(n)
    integer(ip_),  intent(in), optional :: perm(n)
    IF ( control%unit_error >= 0 ) WRITE( control%unit_error,                  &
         "( ' We regret that the solution options that you have ', /,          &
   &     ' chosen are not all freely available with GALAHAD.', /,              &
   &     ' If you have HSL (formerly the Harwell Subroutine', /,               &
   &     ' Library), this option may be enabled by replacing the dummy ', /,   &
   &     ' subroutine MI35_factorize with its HSL namesake ', /,               &
   &     ' and dependencies. See ', /,                                         &
   &     '   $GALAHAD/src/makedefs/packages for details.' )" )
    info%flag = GALAHAD_unavailable_option
    end subroutine mi35_factorize

!****************************************************************************

    subroutine mi35_formC(m,n,ptrA,rowA,valA,ptrC,rowC,valC,                   &
                                 control,info,weight)
    integer(ip_),  intent(in) :: m
    integer(ip_),  intent(in) :: n
    integer(ip_),  intent(in) ::  ptrA(n+1)
    integer(ip_),  intent(in) ::  rowA(:)
    real(rp_), intent(in) ::  valA(:)
    integer(ip_),  intent(out) ::  ptrC(n+1)
    integer(ip_),  intent(out), allocatable ::  rowC(:)
    real(rp_), intent(out), allocatable ::  valC(:)
    type(mi35_control), intent(in) :: control
    type(mi35_info), intent(inout) :: info
    real(rp_), optional, intent(in) ::  weight(m)
    IF ( control%unit_error >= 0 ) WRITE( control%unit_error,                  &
         "( ' We regret that the solution options that you have ', /,          &
   &     ' chosen are not all freely available with GALAHAD.', /,              &
   &     ' If you have HSL (formerly the Harwell Subroutine', /,               &
   &     ' Library), this option may be enabled by replacing the dummy ', /,   &
   &     ' subroutine MI35_formC with its HSL namesake ', /,                   &
   &     ' and dependencies. See ', /,                                         &
   &     '   $GALAHAD/src/makedefs/packages for details.' )" )
    info%flag = GALAHAD_unavailable_option
    end subroutine mi35_formC

!****************************************************************************

    subroutine mi35_factorizeC( n, ptr, row, val, lsize, rsize, keep,          &
                                control, info, scale, perm )
    integer(ip_),  intent(in) :: n
    integer(ip_),  intent(in) ::  ptr(n+1)
    integer(ip_),  intent(in) ::  row(:)
    real(rp_), intent(in) ::  val(:)
    integer(ip_),  intent(in) :: lsize
    integer(ip_),  intent(in) :: rsize
    type(mi35_keep), intent(out) :: keep
    type(mi35_control), intent(in) :: control
    type(mi35_info), intent(inout) :: info
    real(rp_), intent(in), optional :: scale(n)
    integer(ip_),  intent(in), optional :: perm(n)
    IF ( control%unit_error >= 0 ) WRITE( control%unit_error,                  &
         "( ' We regret that the solution options that you have ', /,          &
   &     ' chosen are not all freely available with GALAHAD.', /,              &
   &     ' If you have HSL (formerly the Harwell Subroutine', /,               &
   &     ' Library), this option may be enabled by replacing the dummy ', /,   &
   &     ' subroutine MI35_factorizeC with its HSL namesake ', /,              &
   &     ' and dependencies. See ', /,                                         &
   &     '   $GALAHAD/src/makedefs/packages for details.' )" )
    info%flag = GALAHAD_unavailable_option
    end subroutine mi35_factorizeC

!****************************************************************************

    subroutine mi35_precondition( n, keep, z, y, info )
    integer(ip_),  intent(in) :: n
    type(mi35_keep), intent(inout) :: keep
    real(rp_), intent(in) :: z(n)
    real(rp_), intent(out) :: y(n)
    type(mi35_info), intent(inout) :: info
    info%flag = GALAHAD_unavailable_option
    end subroutine mi35_precondition

!****************************************************************************

    subroutine mi35_solve( trans, n, keep, z, y, info )
    logical(lp_), intent(in) :: trans
    integer(ip_),  intent(in) :: n
    type(mi35_keep), intent(inout) :: keep
    real(rp_), intent(in) :: z(n)
    real(rp_), intent(out) :: y(n)
    type(mi35_info), intent(inout) :: info
    info%flag = GALAHAD_unavailable_option
    end subroutine mi35_solve

!****************************************************************************

    subroutine mi35_finalise( keep, info )
    type(mi35_keep), intent(inout) :: keep
    type(mi35_info), intent(inout) :: info
    info%flag = GALAHAD_unavailable_option
    end subroutine mi35_finalise

  end module hsl_mi35_real
