! THIS VERSION: GALAHAD 5.1 - 2024-10-11 AT 15:00 GMT.

#include "hsl_subset.h"

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M I 2 8   M O D U L E  -*-*-*-

 module hsl_mi28_real
   use hsl_kinds_real, only: ip_, long_, lp_, rp_
#ifdef INTEGER_64
     USE GALAHAD_SYMBOLS_64, ONLY: GALAHAD_unavailable_option
#else
     USE GALAHAD_SYMBOLS, ONLY: GALAHAD_unavailable_option
#endif

   implicit none

   private
   public :: mi28_keep, mi28_control, mi28_info
   public :: mi28_factorize, mi28_finalise, mi28_precondition, mi28_solve
   LOGICAL, PUBLIC, PARAMETER :: mi28_available = .FALSE.

  real(rp_), parameter :: zero = 0.0_rp_
  real(rp_), parameter :: one = 1.0_rp_
  real(rp_), parameter :: sfact = 2.0_rp_
  real(rp_), parameter :: sfact2 = 4.0_rp_
  real(rp_), parameter :: alpham = 0.001_rp_

  interface mi28_factorize
      module procedure mi28_factorize_real
  end interface

  interface mi28_precondition
      module procedure mi28_precondition_real
  end interface

  interface mi28_solve
      module procedure mi28_solve_real
  end interface

  interface mi28_finalise
      module procedure mi28_finalise_real
  end interface

  type mi28_keep
    integer(long_), allocatable ::  fact_ptr(:)
    integer(ip_),  allocatable ::  fact_row(:)
    real(rp_), allocatable ::  fact_val(:)
    real(rp_), allocatable :: scale(:)
    integer(ip_),  allocatable :: invp(:)
    integer(ip_),  allocatable :: perm(:)
    real(rp_), allocatable :: w(:)
  end type mi28_keep

  type mi28_control
    real(rp_) :: alpha = zero
    logical(lp_) :: check = .true.
    integer(ip_) :: iorder = 6
    integer(ip_) :: iscale = 1
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
  end type mi28_control

  type mi28_info
    integer(ip_) :: band_after = 0
    integer(ip_) :: band_before = 0
    integer(ip_) :: dup = 0
    integer(ip_) :: flag = 0
    integer(ip_) :: flag61 = 0
    integer(ip_) :: flag64 = 0
    integer(ip_) :: flag68 = 0
    integer(ip_) :: flag77 = 0
    integer(ip_) :: nrestart = 0
    integer(ip_) :: nshift = 0
    integer(ip_) :: oor = 0
    real(rp_) :: profile_before = 0
    real(rp_) :: profile_after = 0
    integer(long_) :: size_r = 0_long_
    integer(ip_) :: stat = 0
    real(rp_) :: alpha = zero
  end type mi28_info

 contains

  subroutine mi28_factorize_real(n, ptr, row, val, lsize, rsize, keep,       &
      control, info, scale, invp)
    integer(ip_),  intent(in) :: n
    integer(ip_),  intent(inout) ::  ptr(n+1)
    integer(ip_),  intent(inout) ::  row(:)
    real(rp_), intent(inout) ::  val(:)
    integer(ip_),  intent(in) :: lsize
    integer(ip_),  intent(in) :: rsize
    type(mi28_keep), intent(out) :: keep
    type(mi28_control), intent(in) :: control
    type(mi28_info), intent(out) :: info
    real(rp_), intent(in), optional :: scale(n)
    integer(ip_),  intent(in), optional :: invp(n)
    IF ( control%unit_error >= 0 ) WRITE( control%unit_error,                  &
           "( ' We regret that the solution options that you have ', /,        &
   &     ' chosen are not all freely available with GALAHAD.', /,              &
   &     ' If you have HSL (formerly the Harwell Subroutine', /,               &
   &     ' Library), this option may be enabled by replacing the dummy ', /,   &
   &     ' subroutine MI28_factorize with its HSL namesake ', /,               &
   &     ' and dependencies. See ', /,                                         &
   &     '   $GALAHAD/src/makedefs/packages for details.' )" )
    info%flag = GALAHAD_unavailable_option
  end subroutine mi28_factorize_real

  subroutine mi28_precondition_real(n, keep, z, y, info)
    integer(ip_),  intent(in) :: n
    type(mi28_keep), intent(inout) :: keep
    real(rp_), intent(in) :: z(n)
    real(rp_), intent(out) :: y(n)
    type(mi28_info), intent(inout) :: info
    info%flag = GALAHAD_unavailable_option
  end subroutine mi28_precondition_real

  subroutine mi28_solve_real(trans, n, keep, z, y, info)
    logical(lp_), intent(in) :: trans
    integer(ip_),  intent(in) :: n
    type(mi28_keep), intent(inout) :: keep
    real(rp_), intent(in) :: z(n)
    real(rp_), intent(out) :: y(n)
    type(mi28_info), intent(inout) :: info
    info%flag = GALAHAD_unavailable_option

  end subroutine mi28_solve_real

  subroutine mi28_finalise_real(keep, info)
    type(mi28_keep), intent(inout) :: keep
    type(mi28_info), intent(inout) :: info
    info%flag = GALAHAD_unavailable_option
  end subroutine mi28_finalise_real

 end module hsl_mi28_real

