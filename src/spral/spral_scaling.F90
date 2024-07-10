! THIS VERSION: GALAHAD 5.1 - 2024-07-06 AT 14:00 GMT.

#include "spral_procedures.h"

! COPYRIGHT (c) 2014 The Science and Technology Facilities Council (STFC)
! Original date 18 December 2014, Version 1.0.0
!
! Written by: Jonathan Hogg
!
! Hungarian code derives from HSL MC64 code, but has been substantially
! altered for readability and to support rectangular matrices.
! All other code is fresh for SPRAL.
module spral_scaling_precision

  use spral_matrix_util_precision, only : half_to_full
  use spral_kinds_precision
  implicit none

  private
  ! Top level routines
  public :: auction_scale_sym, & ! Symmetric scaling by Auction algorithm
       auction_scale_unsym,    & ! Unsymmetric scaling by Auction algorithm
       equilib_scale_sym,      & ! Sym scaling by Equilibriation (MC77-like)
       equilib_scale_unsym,    & ! Unsym scaling by Equilibriation (MC77-like)
       hungarian_scale_sym,    & ! Sym scaling by Hungarian alg (MC64-like)
       hungarian_scale_unsym     ! Unsym scaling by Hungarian alg (MC64-like)
  ! Inner routines that allow calling internals
  public :: hungarian_match      ! Find a matching (no pre/post-processing)
  ! Data types
  public :: auction_options, auction_inform, &
       equilib_options, equilib_inform,      &
       hungarian_options, hungarian_inform

  real(rp_), parameter :: rinf = huge(rinf)

  type auction_options
     integer(ip_) :: max_iterations = 30000
     integer(ip_) :: max_unchanged(3) = (/ 10,   100, 100 /)
     real :: min_proportion(3) = (/ 0.90, 0.0, 0.0 /)
     real :: eps_initial = 0.01
  end type auction_options

  type auction_inform
     integer(ip_) :: flag = 0 ! success or failure
     integer(ip_) :: stat = 0 ! Fortran stat value on memory allocation failure
     integer(ip_) :: matched = 0 ! #matched rows/cols
     integer(ip_) :: iterations = 0 ! #iterations
     integer(ip_) :: unmatchable = 0 ! #classified as unmatchable
  end type auction_inform

  type equilib_options
     integer(ip_) :: max_iterations = 10
     real :: tol = 1e-8
  end type equilib_options

  type equilib_inform
     integer(ip_) :: flag
     integer(ip_) :: stat
     integer(ip_) :: iterations
  end type equilib_inform

  type hungarian_options
     logical :: scale_if_singular = .false.
  end type hungarian_options

  type hungarian_inform
     integer(ip_) :: flag
     integer(ip_) :: stat
     integer(ip_) :: matched
  end type hungarian_inform

  integer(ip_), parameter :: ERROR_ALLOCATION = -1
  integer(ip_), parameter :: ERROR_SINGULAR = -2

  integer(ip_), parameter :: WARNING_SINGULAR = 1

  interface auction_scale_sym
     module procedure auction_scale_sym_int32
     module procedure auction_scale_sym_int64
  end interface auction_scale_sym
  interface auction_scale_unsym
     module procedure auction_scale_unsym_int32
     module procedure auction_scale_unsym_int64
  end interface auction_scale_unsym
  interface equilib_scale_sym
     module procedure equilib_scale_sym_int32
     module procedure equilib_scale_sym_int64
  end interface equilib_scale_sym
  interface equilib_scale_unsym
     module procedure equilib_scale_unsym_int32
     module procedure equilib_scale_unsym_int64
  end interface equilib_scale_unsym
  interface hungarian_scale_sym
     module procedure hungarian_scale_sym_int32
     module procedure hungarian_scale_sym_int64
  end interface hungarian_scale_sym
  interface hungarian_scale_unsym
     module procedure hungarian_scale_unsym_int32
     module procedure hungarian_scale_unsym_int64
  end interface hungarian_scale_unsym

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Wrappers around scaling algorithms
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!**************************************************************
!
! Use matching-based scaling obtained using Hungarian algorithm (sym)
!
  subroutine hungarian_scale_sym_int32(n, ptr, row, val, scaling, options, &
       inform, match)
    implicit none
    integer(ip_), intent(in) :: n ! order of system
    integer(i4_), intent(in) :: ptr(n+1) ! column pointers of A
    integer(ip_), intent(in) :: row(*) ! row indices of A (lower triangle)
    real(rp_), intent(in) :: val(*) ! entries of A (in same order as in row).
    real(rp_), dimension(n), intent(out) :: scaling
    type(hungarian_options), intent(in) :: options
    type(hungarian_inform), intent(out) :: inform
    integer(ip_), dimension(n), optional, intent(out) :: match

    integer(i8_), dimension(:), allocatable :: ptr64

    allocate(ptr64(n+1), stat=inform%stat)
    if (inform%stat .ne. 0) then
       inform%flag = ERROR_ALLOCATION
       return
    end if
    ptr64(1:n+1) = ptr(1:n+1)

    call hungarian_scale_sym_int64(n, ptr64, row, val, scaling, options, &
         inform, match=match)
  end subroutine hungarian_scale_sym_int32

  subroutine hungarian_scale_sym_int64(n, ptr, row, val, scaling, options, &
       inform, match)
    implicit none
    integer(ip_), intent(in) :: n ! order of system
    integer(i8_), intent(in) :: ptr(n+1) ! column pointers of A
    integer(ip_), intent(in) :: row(*) ! row indices of A (lower triangle)
    real(rp_), intent(in) :: val(*) ! entries of A (in same order as in row).
    real(rp_), dimension(n), intent(out) :: scaling
    type(hungarian_options), intent(in) :: options
    type(hungarian_inform), intent(out) :: inform
    integer(ip_), dimension(n), optional, intent(out) :: match

    integer(ip_), dimension(:), allocatable :: perm
    real(rp_), dimension(:), allocatable :: rscaling, cscaling

    inform%flag = 0 ! Initialize to success

    allocate(rscaling(n), cscaling(n), stat=inform%stat)
    if (inform%stat .ne. 0) then
       inform%flag = ERROR_ALLOCATION
       return
    end if

    if (present(match)) then
       call hungarian_wrapper(.true., n, n, ptr, row, val, match, rscaling, &
            cscaling, options, inform)
    else
       allocate(perm(n), stat=inform%stat)
       if (inform%stat .ne. 0) then
          inform%flag = ERROR_ALLOCATION
          return
       end if
       call hungarian_wrapper(.true., n, n, ptr, row, val, perm, rscaling, &
            cscaling, options, inform)
    end if
    scaling(1:n) = exp( (rscaling(1:n) + cscaling(1:n)) / 2 )
  end subroutine hungarian_scale_sym_int64

!**************************************************************
!
! Use matching-based scaling obtained using Hungarian algorithm (unsym)
!
  subroutine hungarian_scale_unsym_int32(m, n, ptr, row, val, rscaling, &
       cscaling, options, inform, match)
    implicit none
    integer(ip_), intent(in) :: m ! number of rows
    integer(ip_), intent(in) :: n ! number of cols
    integer(i4_), intent(in) :: ptr(n+1) ! column pointers of A
    integer(ip_), intent(in) :: row(*) ! row indices of A (lower triangle)
    real(rp_), intent(in) :: val(*) ! entries of A (in same order as in row).
    real(rp_), dimension(m), intent(out) :: rscaling
    real(rp_), dimension(n), intent(out) :: cscaling
    type(hungarian_options), intent(in) :: options
    type(hungarian_inform), intent(out) :: inform
    integer(ip_), dimension(m), optional, intent(out) :: match

    integer(i8_), dimension(:), allocatable :: ptr64

    ! Copy from int32 to int64
    allocate(ptr64(n+1), stat=inform%stat)
    if (inform%stat .ne. 0) then
       inform%flag = ERROR_ALLOCATION
       return
    end if
    ptr64(1:n+1) = ptr(1:n+1)

    call hungarian_scale_unsym_int64(m, n, ptr64, row, val, rscaling, &
         cscaling, options, inform, match=match)
  end subroutine hungarian_scale_unsym_int32

  subroutine hungarian_scale_unsym_int64(m, n, ptr, row, val, rscaling, &
         cscaling, options, inform, match)
    implicit none
    integer(ip_), intent(in) :: m ! number of rows
    integer(ip_), intent(in) :: n ! number of cols
    integer(i8_), intent(in) :: ptr(n+1) ! column pointers of A
    integer(ip_), intent(in) :: row(*) ! row indices of A (lower triangle)
    real(rp_), intent(in) :: val(*) ! entries of A (in same order as in row).
    real(rp_), dimension(m), intent(out) :: rscaling
    real(rp_), dimension(n), intent(out) :: cscaling
    type(hungarian_options), intent(in) :: options
    type(hungarian_inform), intent(out) :: inform
    integer(ip_), dimension(m), optional, intent(out) :: match

    integer(ip_), dimension(:), allocatable :: perm

    inform%flag = 0 ! Initialize to success

    ! Call main routine
    if (present(match)) then
       call hungarian_wrapper(.false., m, n, ptr, row, val, match, rscaling, &
            cscaling, options, inform)
    else
       allocate(perm(m), stat=inform%stat)
       if (inform%stat .ne. 0) then
          inform%flag = ERROR_ALLOCATION
          return
       end if
       call hungarian_wrapper(.false., m, n, ptr, row, val, perm, rscaling, &
            cscaling, options, inform)
    end if

    ! Apply post processing
    rscaling(1:m) = exp( rscaling(1:m) )
    cscaling(1:n) = exp( cscaling(1:n) )
  end subroutine hungarian_scale_unsym_int64

!**************************************************************
!
! Call auction algorithm to get a scaling, then symmetrize it
!
  subroutine auction_scale_sym_int32(n, ptr, row, val, scaling, options, &
       inform, match)
    implicit none
    integer(ip_), intent(in) :: n ! order of system
    integer(i4_), intent(in) :: ptr(n+1) ! column pointers of A
    integer(ip_), intent(in) :: row(*) ! row indices of A (lower triangle)
    real(rp_), intent(in) :: val(*) ! entries of A (in same order as in row).
    real(rp_), dimension(n), intent(out) :: scaling
    type(auction_options), intent(in) :: options
    type(auction_inform), intent(out) :: inform
    integer(ip_), dimension(n), optional, intent(out) :: match

    integer(i8_), dimension(:), allocatable :: ptr64

    allocate(ptr64(n+1), stat=inform%stat)
    if (inform%stat .ne. 0) then
       inform%flag = ERROR_ALLOCATION
       return
    end if
    ptr64(1:n+1) = ptr(1:n+1)

    call auction_scale_sym_int64(n, ptr64, row, val, scaling, options, inform, &
         match=match)
  end subroutine auction_scale_sym_int32

  subroutine auction_scale_sym_int64(n, ptr, row, val, scaling, options, &
       inform, match)
    implicit none
    integer(ip_), intent(in) :: n ! order of system
    integer(i8_), intent(in) :: ptr(n+1) ! column pointers of A
    integer(ip_), intent(in) :: row(*) ! row indices of A (lower triangle)
    real(rp_), intent(in) :: val(*) ! entries of A (in same order as in row).
    real(rp_), dimension(n), intent(out) :: scaling
    type(auction_options), intent(in) :: options
    type(auction_inform), intent(out) :: inform
    integer(ip_), dimension(n), optional, intent(out) :: match

    integer(ip_), dimension(:), allocatable :: perm
    real(rp_), dimension(:), allocatable :: rscaling, cscaling

    inform%flag = 0 ! Initialize to sucess

    ! Allocate memory
    allocate(rscaling(n), cscaling(n), stat=inform%stat)
    if (inform%stat .ne. 0) then
       inform%flag = ERROR_ALLOCATION
       return
    end if

    ! Call unsymmetric implementation with flag to expand half matrix
    if (present(match)) then
       call auction_match(.true., n, n, ptr, row, val, match, rscaling, &
            cscaling, options, inform)
    else
       allocate(perm(n), stat=inform%stat)
       if (inform%stat .ne. 0) then
          inform%flag = ERROR_ALLOCATION
          return
       end if
       call auction_match(.true., n, n, ptr, row, val, perm, rscaling, &
            cscaling, options, inform)
    end if

    ! Average rscaling and cscaling to get symmetric scaling
    scaling(1:n) = exp( (rscaling(1:n)+cscaling(1:n))/2 )
  end subroutine auction_scale_sym_int64

!**************************************************************
!
! Call auction algorithm to get a scaling (unsymmetric version)
!
  subroutine auction_scale_unsym_int32(m, n, ptr, row, val, rscaling, &
       cscaling, options, inform, match)
    implicit none
    integer(ip_), intent(in) :: m ! number of rows
    integer(ip_), intent(in) :: n ! number of columns
    integer(i4_), intent(in) :: ptr(n+1) ! column pointers of A
    integer(ip_), intent(in) :: row(*) ! row indices of A (lower triangle)
    real(rp_), intent(in) :: val(*) ! entries of A (in same order as in row).
    real(rp_), dimension(m), intent(out) :: rscaling
    real(rp_), dimension(n), intent(out) :: cscaling
    type(auction_options), intent(in) :: options
    type(auction_inform), intent(out) :: inform
    integer(ip_), dimension(m), optional, intent(out) :: match

    integer(i8_), dimension(:), allocatable :: ptr64

    allocate(ptr64(n+1), stat=inform%stat)
    if (inform%stat .ne. 0) then
       inform%flag = ERROR_ALLOCATION
       return
    end if
    ptr64(1:n+1) = ptr(1:n+1)

    call auction_scale_unsym_int64(m, n, ptr64, row, val, rscaling, cscaling, &
         options, inform, match=match)
  end subroutine auction_scale_unsym_int32

  subroutine auction_scale_unsym_int64(m, n, ptr, row, val, rscaling, &
       cscaling, options, inform, match)
    implicit none
    integer(ip_), intent(in) :: m ! number of rows
    integer(ip_), intent(in) :: n ! number of columns
    integer(i8_), intent(in) :: ptr(n+1) ! column pointers of A
    integer(ip_), intent(in) :: row(*) ! row indices of A (lower triangle)
    real(rp_), intent(in) :: val(*) ! entries of A (in same order as in row).
    real(rp_), dimension(m), intent(out) :: rscaling
    real(rp_), dimension(n), intent(out) :: cscaling
    type(auction_options), intent(in) :: options
    type(auction_inform), intent(out) :: inform
    integer(ip_), dimension(m), optional, intent(out) :: match

    integer(ip_), dimension(:), allocatable :: perm

    inform%flag = 0 ! Initialize to sucess

    if (present(match)) then
       call auction_match(.false., m, n, ptr, row, val, match, rscaling, &
            cscaling, options, inform)
    else
       allocate(perm(m), stat=inform%stat)
       if (inform%stat .ne. 0) then
          inform%flag = ERROR_ALLOCATION
          return
       end if
       call auction_match(.false., m, n, ptr, row, val, perm, rscaling, &
            cscaling, options, inform)
    end if

    rscaling(1:m) = exp(rscaling(1:m))
    cscaling(1:n) = exp(cscaling(1:n))
  end subroutine auction_scale_unsym_int64

!*******************************
!
! Call the infinity-norm equilibriation algorithm (symmetric version)
!
  subroutine equilib_scale_sym_int32(n, ptr, row, val, scaling, options, inform)
    implicit none
    integer(ip_), intent(in) :: n ! order of system
    integer(i4_), intent(in) :: ptr(n+1) ! column pointers of A
    integer(ip_), intent(in) :: row(*) ! row indices of A (lower triangle)
    real(rp_), intent(in) :: val(*) ! entries of A (in same order as in row).
    real(rp_), dimension(n), intent(out) :: scaling
    type(equilib_options), intent(in) :: options
    type(equilib_inform), intent(out) :: inform

    integer(i8_), dimension(:), allocatable :: ptr64

    allocate(ptr64(n+1), stat=inform%stat)
    if (inform%stat .ne. 0) then
       inform%flag = ERROR_ALLOCATION
       return
    end if
    ptr64(1:n+1) = ptr(1:n+1)

    call equilib_scale_sym_int64(n, ptr64, row, val, scaling, options, inform)
  end subroutine equilib_scale_sym_int32

  subroutine equilib_scale_sym_int64(n, ptr, row, val, scaling, options, inform)
    implicit none
    integer(ip_), intent(in) :: n ! order of system
    integer(i8_), intent(in) :: ptr(n+1) ! column pointers of A
    integer(ip_), intent(in) :: row(*) ! row indices of A (lower triangle)
    real(rp_), intent(in) :: val(*) ! entries of A (in same order as in row).
    real(rp_), dimension(n), intent(out) :: scaling
    type(equilib_options), intent(in) :: options
    type(equilib_inform), intent(out) :: inform

    inform%flag = 0 ! Initialize to sucess

    call inf_norm_equilib_sym(n, ptr, row, val, scaling, options, inform)
  end subroutine equilib_scale_sym_int64

!*******************************
!
! Call the infinity-norm equilibriation algorithm (unsymmetric version)
!
  subroutine equilib_scale_unsym_int32(m, n, ptr, row, val, rscaling, &
       cscaling, options, inform)
    implicit none
    integer(ip_), intent(in) :: m ! number of rows
    integer(ip_), intent(in) :: n ! number of cols
    integer(i4_), intent(in) :: ptr(n+1) ! column pointers of A
    integer(ip_), intent(in) :: row(*) ! row indices of A (lower triangle)
    real(rp_), intent(in) :: val(*) ! entries of A (in same order as in row).
    real(rp_), dimension(m), intent(out) :: rscaling
    real(rp_), dimension(n), intent(out) :: cscaling
    type(equilib_options), intent(in) :: options
    type(equilib_inform), intent(out) :: inform

    integer(i8_), dimension(:), allocatable :: ptr64

    allocate(ptr64(n+1), stat=inform%stat)
    if (inform%stat .ne. 0) then
       inform%flag = ERROR_ALLOCATION
       return
    end if
    ptr64(1:n+1) = ptr(1:n+1)

    call equilib_scale_unsym_int64(m, n, ptr64, row, val, rscaling, cscaling, &
         options, inform)
  end subroutine equilib_scale_unsym_int32

  subroutine equilib_scale_unsym_int64(m, n, ptr, row, val, rscaling, &
       cscaling, options, inform)
    implicit none
    integer(ip_), intent(in) :: m ! number of rows
    integer(ip_), intent(in) :: n ! number of cols
    integer(i8_), intent(in) :: ptr(n+1) ! column pointers of A
    integer(ip_), intent(in) :: row(*) ! row indices of A (lower triangle)
    real(rp_), intent(in) :: val(*) ! entries of A (in same order as in row).
    real(rp_), dimension(m), intent(out) :: rscaling
    real(rp_), dimension(n), intent(out) :: cscaling
    type(equilib_options), intent(in) :: options
    type(equilib_inform), intent(out) :: inform

    inform%flag = 0 ! Initialize to sucess

    call inf_norm_equilib_unsym(m, n, ptr, row, val, rscaling, cscaling, &
         options, inform)
  end subroutine equilib_scale_unsym_int64

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Inf-norm Equilibriation Algorithm Implementation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!
! We implement Algorithm 1 of:
! A Symmetry Preserving Algorithm for Matrix Scaling
! Philip Knight, Daniel Ruiz and Bora Ucar
! INRIA Research Report 7552 (Novemeber 2012)
! (This is similar to the algorithm used in MC77, but is a complete
!  reimplementation from the above paper to ensure it is 100% STFC
!  copyright and can be released as open source)
!
  subroutine inf_norm_equilib_sym(n, ptr, row, val, scaling, options, inform)
    implicit none
    integer(ip_), intent(in) :: n
    integer(long_), dimension(n+1), intent(in) :: ptr
    integer(ip_), dimension(ptr(n+1)-1), intent(in) :: row
    real(rp_), dimension(ptr(n+1)-1), intent(in) :: val
    real(rp_), dimension(n), intent(out) :: scaling
    type(equilib_options), intent(in) :: options
    type(equilib_inform), intent(inout) :: inform

    integer(ip_) :: itr, r, c
    integer(long_) :: j
    real(rp_) :: v
    real(rp_), dimension(:), allocatable :: maxentry

    allocate(maxentry(n), stat=inform%stat)
    if (inform%stat .ne. 0) then
       inform%flag = ERROR_ALLOCATION
       return
    end if

    scaling(1:n) = 1.0
    do itr = 1, options%max_iterations
       ! Find maximum entry in each row and col
       ! Recall: matrix is symmetric, but we only have half
       maxentry(1:n) = 0.0
       do c = 1, n
          do j = ptr(c), ptr(c+1)-1
             r = row(j)
             v = abs(scaling(r) * val(j) * scaling(c))
             maxentry(r) = max(maxentry(r), v)
             maxentry(c) = max(maxentry(c), v)
          end do
       end do
       ! Update scaling (but beware empty cols)
       where (maxentry(1:n) .gt. 0) &
            scaling(1:n) = scaling(1:n) / sqrt(maxentry(1:n))
       ! Test convergence
       if (maxval(abs(1-maxentry(1:n))) .lt. options%tol) exit
    end do
    inform%iterations = itr-1
  end subroutine inf_norm_equilib_sym

!
! We implement Algorithm 1 of:
! A Symmetry Preserving Algorithm for Matrix Scaling
! Philip Knight, Daniel Ruiz and Bora Ucar
! INRIA Research Report 7552 (Novemeber 2012)
! (This is similar to the algorithm used in MC77, but is a complete
!  reimplementation from the above paper to ensure it is 100% STFC
!  copyright and can be released as open source)
!
  subroutine inf_norm_equilib_unsym(m, n, ptr, row, val, rscaling, cscaling, &
       options, inform)
    implicit none
    integer(ip_), intent(in) :: m
    integer(ip_), intent(in) :: n
    integer(long_), dimension(n+1), intent(in) :: ptr
    integer(ip_), dimension(ptr(n+1)-1), intent(in) :: row
    real(rp_), dimension(ptr(n+1)-1), intent(in) :: val
    real(rp_), dimension(m), intent(out) :: rscaling
    real(rp_), dimension(n), intent(out) :: cscaling
    type(equilib_options), intent(in) :: options
    type(equilib_inform), intent(inout) :: inform

    integer(ip_) :: itr, r, c
    integer(long_) :: j
    real(rp_) :: v
    real(rp_), dimension(:), allocatable :: rmaxentry, cmaxentry

    allocate(rmaxentry(m), cmaxentry(n), stat=inform%stat)
    if (inform%stat .ne. 0) then
       inform%flag = ERROR_ALLOCATION
       return
    end if

    rscaling(1:m) = 1.0
    cscaling(1:n) = 1.0
    do itr = 1, options%max_iterations
       ! Find maximum entry in each row and col
       ! Recall: matrix is symmetric, but we only have half
       rmaxentry(1:m) = 0.0
       cmaxentry(1:n) = 0.0
       do c = 1, n
          do j = ptr(c), ptr(c+1)-1
             r = row(j)
             v = abs(rscaling(r) * val(j) * cscaling(c))
             rmaxentry(r) = max(rmaxentry(r), v)
             cmaxentry(c) = max(cmaxentry(c), v)
          end do
       end do
       ! Update scaling (but beware empty cols)
       where(rmaxentry(1:m).gt.0) &
            rscaling(1:m) = rscaling(1:m) / sqrt(rmaxentry(1:m))
       where(cmaxentry(1:n).gt.0) &
            cscaling(1:n) = cscaling(1:n) / sqrt(cmaxentry(1:n))
       ! Test convergence
       if ((maxval(abs(1-rmaxentry(1:m))) .lt. options%tol) .and. &
            (maxval(abs(1-cmaxentry(1:n))) .lt. options%tol)) exit
    end do
    inform%iterations = itr-1
  end subroutine inf_norm_equilib_unsym

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Hungarian Algorithm implementation (MC64)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!**********************************************************************
!
! This subroutine wraps the core algorithm of hungarian_match(). It provides
! pre- and post-processing to transform a maximum product assignment to a
! minimum sum assignment problem (and back again). It also has post-processing
! to handle the case of a structurally singular matrix as per Duff and Pralet
! (though the efficacy of such an approach is disputed!)
!
! This code is adapted from HSL_MC64 v2.3.1
!
  subroutine hungarian_wrapper(sym, m, n, ptr, row, val, match, rscaling, &
       cscaling, options, inform)
    implicit none
    logical, intent(in) :: sym
    integer(ip_), intent(in) :: m
    integer(ip_), intent(in) :: n
    integer(long_), dimension(n+1), intent(in) :: ptr
    integer(ip_), dimension(*), intent(in) :: row
    real(rp_), dimension(*), intent(in) :: val
    integer(ip_), dimension(m), intent(out) :: match
    real(rp_), dimension(m), intent(out) :: rscaling
    real(rp_), dimension(n), intent(out) :: cscaling
    type(hungarian_options), intent(in) :: options
    type(hungarian_inform), intent(out) :: inform

    integer(long_), allocatable :: ptr2(:)
    integer(ip_), allocatable :: row2(:), iw(:), new_to_old(:), &
         old_to_new(:), cperm(:)
    real(rp_), allocatable :: val2(:), dualu(:), dualv(:), cmax(:), cscale(:)
    real(rp_) :: colmax
    integer(ip_) :: i, j, nn, jj, k
    integer(long_) :: j1, j2, jlong, klong, ne
    real(rp_), parameter :: zero = 0.0

    inform%flag = 0
    inform%stat = 0
    ne = ptr(n+1)-1

    ! Reset ne for the expanded symmetric matrix
    ne = 2*ne

    ! Expand matrix, drop explicit zeroes and take log absolute values
    allocate(ptr2(n+1), row2(ne), val2(ne), &
         iw(5*n), dualu(m), dualv(n), cmax(n), stat=inform%stat)
    if (inform%stat .ne. 0) then
       inform%flag = ERROR_ALLOCATION
       return
    end if

    klong = 1
    do i = 1, n
       ptr2(i) = klong
       do jlong = ptr(i), ptr(i+1)-1
          if (val(jlong) .eq. zero) cycle
          row2(klong) = row(jlong)
          val2(klong) = abs(val(jlong))
          klong = klong + 1
       end do
       ! Following log is seperated from above loop to expose expensive
       ! log operation to vectorization.
       val2(ptr2(i):klong-1) = log(val2(ptr2(i):klong-1))
    end do
    ptr2(n+1) = klong
    if (sym) call half_to_full(n, row2, ptr2, iw, a=val2)

    ! Compute column maximums
    do i = 1, n
       colmax = maxval(val2(ptr2(i):ptr2(i+1)-1))
       cmax(i) = colmax
       val2(ptr2(i):ptr2(i+1)-1) = colmax - val2(ptr2(i):ptr2(i+1)-1)
    end do

    call hungarian_match(m, n, ptr2, row2, val2, match, inform%matched, dualu, &
         dualv, inform%stat)
    if (inform%stat .ne. 0) then
       inform%flag = ERROR_ALLOCATION
       return
    end if

    if (inform%matched .ne. min(m,n)) then
       ! Singular matrix
       if (options%scale_if_singular) then
          ! Just issue warning then continue
          inform%flag = WARNING_SINGULAR
       else
          ! Issue error and return identity scaling
          inform%flag = ERROR_SINGULAR
          rscaling(1:m) = 0
          cscaling(1:n) = 0
       end if
    end if

    if ((.not. sym) .or. (inform%matched .eq. n)) then
       ! Unsymmetric or symmetric and full rank. Note that in this case m=n
       rscaling(1:m) = dualu(1:m)
       cscaling(1:n) = dualv(1:n) - cmax(1:n)
       call match_postproc(m, n, ptr, row, val, rscaling, cscaling, &
            inform%matched, match, inform%flag, inform%stat)
       return
    end if

    ! If we reach this point then structually rank deficient.
    ! As matching may not involve full set of rows and columns, but we need
    ! a symmetric matching/scaling, we can't just return the current matching.
    ! Instead, we build a full rank submatrix and call matching on it.

    allocate(old_to_new(n),new_to_old(n),cperm(n),stat=inform%stat)
    if (inform%stat .ne. 0) then
       inform%flag = ERROR_ALLOCATION
       return
    end if

    j = inform%matched + 1
    k = 0
    do i = 1,m
       if (match(i) < 0) then
          ! row i is not part of the matching
          old_to_new(i) = -j
          j = j + 1
       else
          k = k + 1
          ! old_to_new(i) holds the new index for variable i after
          ! removal of singular part and new_to_old(k) is the
          ! original index for k
          old_to_new(i) = k
          new_to_old(k) = i
       end if
    end do

    ! Overwrite ptr2, row2 and val2
    ne = 0
    k = 0
    ptr2(1) = 1
    j2 = 1
    do i = 1,n
       j1 = j2
       j2 = ptr2(i+1)
       ! skip over unmatched entries
       if (match(i) < 0) cycle
       k = k + 1
       do jlong = j1,j2-1
          jj = row2(jlong)
          if (match(jj) < 0) cycle
          ne = ne + 1
          row2(ne) = old_to_new(jj)
          val2(ne) = val2(jlong)
       end do
       ptr2(k+1) = ne + 1
    end do
    ! nn is order of non-singular part.
    nn = k
    call hungarian_match(nn, nn, ptr2, row2, val2, cperm, inform%matched, &
         dualu, dualv, inform%stat)
    if (inform%stat .ne. 0) then
       inform%flag = ERROR_ALLOCATION
       return
    end if

    do i = 1,n
       j = old_to_new(i)
       if (j < 0) then
          rscaling(i) = -huge(rscaling)
       else
          ! Note: we need to subtract col max using old matrix numbering
          rscaling(i) = (dualu(j)+dualv(j)-cmax(i))/2
       end if
    end do

    match(1:n) = -1
    do i = 1,nn
       j = cperm(i)
       match(new_to_old(i)) = j
    end do

    do i = 1, n
       if (match(i) .eq. -1) then
          match(i) = old_to_new(i)
       end if
    end do

    ! Apply Duff and Pralet correction to unmatched row scalings
    allocate(cscale(n), stat=inform%stat)
    if (inform%stat .ne. 0) then
       inform%flag = ERROR_ALLOCATION
       return
    end if
    ! For columns i not in the matched set I, set
    !     s_i = 1 / (max_{k in I} | a_ik s_k |)
    ! with convention that 1/0 = 1
    cscale(1:n) = rscaling(1:n)
    do i = 1,n
       do jlong = ptr(i), ptr(i+1)-1
          k = row(jlong)
          if ((cscale(i) .eq. -huge(rscaling)) .and. &
              (cscale(k) .ne. -huge(rscaling))) then
             ! i not in I, k in I
             rscaling(i) = max(rscaling(i), log(abs(val(jlong)))+rscaling(k))
          end if
          if ((cscale(k) .eq. -huge(rscaling)) .and. &
              (cscale(i) .ne. -huge(rscaling))) then
             ! k not in I, i in I
             rscaling(k) = max(rscaling(k), log(abs(val(jlong)))+rscaling(i))
          end if
       end do
    end do
    do i = 1,n
       if (cscale(i) .ne. -huge(rscaling)) cycle ! matched part
       if (rscaling(i) .eq. -huge(rscaling)) then
          rscaling(i) = 0.0
       else
          rscaling(i) = -rscaling(i)
       end if
    end do
    ! As symmetric, scaling is averaged on return, but rscaling(:) is correct,
    ! so just copy to cscaling to fix this
    cscaling(1:n) = rscaling(1:n)
  end subroutine hungarian_wrapper

!**********************************************************************
!
! Subroutine that initialize matching and (row) dual variable into a suitbale
! state for main Hungarian algorithm.
!
! The heuristic guaruntees that the generated partial matching is optimal
! on the restriction of the graph to the matched rows and columns.
  subroutine hungarian_init_heurisitic(m, n, ptr, row, val, num, iperm, jperm, &
       dualu, d, l, search_from)
    implicit none
    integer(ip_), intent(in) :: m
    integer(ip_), intent(in) :: n
    integer(long_), dimension(n+1), intent(in) :: ptr
    integer(ip_), dimension(ptr(n+1)-1), intent(in) :: row
    real(rp_), dimension(ptr(n+1)-1), intent(in) :: val
    integer(ip_), intent(inout) :: num
    integer(ip_), dimension(*), intent(inout) :: iperm
    integer(long_), dimension(*), intent(inout) :: jperm
    real(rp_), dimension(m), intent(out) :: dualu
    real(rp_), dimension(n), intent(out) :: d ! d(j) current improvement from
      ! matching in col j
    integer(long_), dimension(m), intent(out) :: l ! position of smallest entry
      ! of row
    integer(long_), dimension(n), intent(inout) :: search_from ! position we
      ! have reached in current search

    integer(ip_) :: i, i0, ii, j, jj
    integer(long_) :: k, k0, kk
    real(rp_) :: di, vj

    !
    ! Set up initial matching on smallest entry in each row (as far as possible)
    !
    ! Find smallest entry in each col, and record it
    dualu(1:m) = RINF
    l(1:m) = 0
    do j = 1, n
       do k = ptr(j),ptr(j+1)-1
          i = row(k)
          if (val(k) .gt. dualu(i)) cycle
          dualu(i) = val(k) ! Initialize dual variables
          iperm(i) = j      ! Record col
          l(i) = k          ! Record posn in row(:)
       end do
    end do
    ! Loop over rows in turn. If we can match on smallest entry in row (i.e.
    ! column not already matched) then do so. Avoid matching on dense columns
    ! as this makes Hungarian algorithm take longer.
    do i = 1, m
       j = iperm(i) ! Smallest entry in row i is (i,j)
       if (j .eq. 0) cycle ! skip empty rows
       iperm(i) = 0
       if (jperm(j) .ne. 0) cycle ! If we've already matched column j, skip
       ! this row. Don't choose cheap assignment from dense columns
       if ((ptr(j+1)-ptr(j) .gt. m/10) .and. (m .gt. 50)) cycle
       ! Assignment of column j to row i
       num = num + 1
       iperm(i) = j
       jperm(j) = l(i)
    end do
    ! If we already have a complete matching, we're already done!
    if (num .eq. min(m,n)) return

    !
    ! Scan unassigned columns; improve assignment
    !
    d(1:n) = 0.0
    search_from(1:n) = ptr(1:n)
    improve_assign: do j = 1, n
       if (jperm(j) .ne. 0) cycle ! column j already matched
       if (ptr(j) .gt. (ptr(j+1)-1)) cycle ! column j is empty
       ! Find smallest value of di = a_ij - u_i in column j
       ! In case of a tie, prefer first unmatched, then first matched row.
       i0 = row(ptr(j))
       vj = val(ptr(j)) - dualu(i0)
       k0 = ptr(j)
       do k = ptr(j)+1, ptr(j+1)-1
          i = row(k)
          di = val(k) - dualu(i)
          if (di .gt. vj) cycle
          if ((di .eq. vj) .and. (di .ne. RINF)) then
             if ((iperm(i) .ne. 0) .or. (iperm(i0) .eq. 0)) cycle
          end if
          vj = di
          i0 = i
          k0 = k
       end do
       ! Record value of matching on (i0,j)
       d(j) = vj
       ! If row i is unmatched, then match on (i0,j) immediately
       if (iperm(i0) .eq. 0) then
          num = num + 1
          jperm(j) = k0
          iperm(i0) = j
          search_from(j) = k0 + 1
          cycle
       end if
       ! Otherwise, row i is matched. Consider all rows i in column j that tie
       ! for this vj value. Such a row currently matches on (i,jj). Scan column
       ! jj looking for an unmatched row ii that improves value of matching. If
       ! one exists, then augment along length 2 path (i,j)->(ii,jj)
       do k = k0, ptr(j+1)-1
          i = row(k)
          if ((val(k)-dualu(i)) .gt. vj) cycle ! Not a tie for vj value
          jj = iperm(i)
          ! Scan remaining part of assigned column jj
          do kk = search_from(jj), ptr(jj+1)-1
             ii = row(kk)
             if (iperm(ii) .gt. 0) cycle ! row ii already matched
             if ((val(kk)-dualu(ii)) .le. d(jj)) then
                ! By matching on (i,j) and (ii,jj) we do better than existing
                ! matching on (i,jj) alone.
                jperm(jj) = kk
                iperm(ii) = jj
                search_from(jj) = kk + 1
                num = num + 1
                jperm(j) = k
                iperm(i) = j
                search_from(j) = k + 1
                cycle improve_assign
             end if
          end do
          search_from(jj) = ptr(jj+1)
       end do
       cycle
    end do improve_assign
  end subroutine hungarian_init_heurisitic

!**********************************************************************
!
! Provides the core Hungarian Algorithm implementation for solving the
! minimum sum assignment problem as per Duff and Koster.
!
! This code is adapted from MC64 v 1.6.0
!
  subroutine hungarian_match(m,n,ptr,row,val,iperm,num,dualu,dualv,st)
    implicit none
    integer(ip_), intent(in) :: m ! number of rows
    integer(ip_), intent(in) :: n ! number of cols
    integer(ip_), intent(out) :: num ! cardinality of the matching
    integer(long_), intent(in) :: ptr(n+1) ! column pointers
    integer(ip_), intent(in) :: row(ptr(n+1)-1) ! row pointers
    integer(ip_), intent(out) :: iperm(m) ! matching itself: row i is matched to
      ! column iperm(i).
    real(rp_), intent(in) :: val(ptr(n+1)-1) ! value of the entry that
      ! corresponds to row(k). All values val(k) must be non-negative.
    real(rp_), intent(out) :: dualu(m) ! dualu(i) is the reduced weight 4 row(i)
    real(rp_), intent(out) :: dualv(n) ! dualv(j) is the reduced weight 4 col(j)
    integer(ip_), intent(out) :: st

    integer(long_), allocatable, dimension(:) :: jperm ! a(jperm(j)) is entry of
      ! A for matching in column j.
    integer(long_), allocatable, dimension(:) :: out ! a(out(i)) is the new
      ! entry in a on which we match going along the short path back to
      ! original col.
    integer(ip_), allocatable, dimension(:) :: pr ! pr(i) is a pointer to the
      ! next column along the shortest path back to the original column
    integer(ip_), allocatable, dimension(:) :: q ! q(1:qlen) forms a binary heap
      ! data structure sorted by d(q(i)) value. q(low:up) is a list of rows
      ! with equal d(i) which is lower or equal to smallest in the heap.
      ! q(up:n) is a list of already visited rows.
    integer(long_), allocatable, dimension(:) :: longwork
    integer(ip_), allocatable, dimension(:) :: l ! l(:) is an inverse of q(:)
    real(rp_), allocatable, dimension(:) :: d ! d(i) is current shortest
      ! distance to row i from current column (d_i from Fig 4.1 of Duff and
      ! Koster paper)

    integer(ip_) :: i,j,jj,jord,q0,qlen,jdum,jsp
    integer(ip_) :: kk,up,low,lpos
    integer(long_) :: klong, isp
    real(rp_) :: csp,di,dmin,dnew,dq0,vj

    !
    ! Initialization
    !
    allocate(jperm(n), out(n), pr(n), q(m), longwork(m), l(m), d(max(m,n)), &
         stat=st)
    if (st .ne. 0) return
    num = 0
    iperm(1:m) = 0
    jperm(1:n) = 0

    call hungarian_init_heurisitic(m, n, ptr, row, val, num, iperm, jperm, &
         dualu, d, longwork, out)
    if (num .eq. min(m,n)) go to 1000 ! If we got a complete matching, done

    !
    ! Repeatedly find augmenting paths until all columns are included in the
    ! matching. At every step the current matching is optimal on the restriction
    ! of the graph to currently matched rows and columns.
    !

    ! Main loop ... each pass round this loop is similar to Dijkstra's
    ! algorithm for solving the single source shortest path problem
    d(1:m) = RINF
    l(1:m) = 0
    isp=-1; jsp=-1 ! initalize to avoid "may be used unitialized" warning
    do jord = 1, n

       if (jperm(jord) .ne. 0) cycle
       ! jord is next unmatched column
       ! dmin is the length of shortest path in the tree
       dmin = RINF
       qlen = 0
       low = m + 1
       up = m + 1
       ! csp is the cost of the shortest augmenting path to unassigned row
       ! row(isp). The corresponding column index is jsp.
       csp = RINF
       ! Build shortest path tree starting from unassigned column (root) jord
       j = jord
       pr(j) = -1

       ! Scan column j
       do klong = ptr(j), ptr(j+1)-1
          i = row(klong)
          dnew = val(klong) - dualu(i)
          if (dnew .ge. csp) cycle
          if (iperm(i) .eq. 0) then
             csp = dnew
             isp = klong
             jsp = j
          else
             if (dnew .lt. dmin) dmin = dnew
             d(i) = dnew
             qlen = qlen + 1
             longwork(qlen) = klong
          end if
       end do
       ! Initialize heap Q and Q2 with rows held in longwork(1:qlen)
       q0 = qlen
       qlen = 0
       do kk = 1, q0
          klong = longwork(kk)
          i = row(klong)
          if (csp .le. d(i)) then
             d(i) = RINF
             cycle
          end if
          if (d(i) .le. dmin) then
             low = low - 1
             q(low) = i
             l(i) = low
          else
             qlen = qlen + 1
             l(i) = qlen
             call heap_update(i,m,Q,D,L)
          end if
          ! Update tree
          jj = iperm(i)
          out(jj) = klong
          pr(jj) = j
       end do

       do jdum = 1,num
          ! If Q2 is empty, extract rows from Q
          if (low .eq. up) then
             if (qlen .eq. 0) exit
             i = q(1) ! Peek at top of heap
             if (d(i) .ge. csp) exit
             dmin = d(i)
             ! Extract all paths that have length dmin and store in q(low:up-1)
             do while (qlen .gt. 0)
                i = q(1) ! Peek at top of heap
                if (d(i) .gt. dmin) exit
                i = heap_pop(qlen,m,Q,D,L)
                low = low - 1
                q(low) = i
                l(i) = low
             end do
          end if
          ! q0 is row whose distance d(q0) to the root is smallest
          q0 = q(up-1)
          dq0 = d(q0)
          ! Exit loop if path to q0 is longer than the shortest augmenting path
          if (dq0 .ge. csp) exit
          up = up - 1

          ! Scan column that matches with row q0
          j = iperm(q0)
          vj = dq0 - val(jperm(j)) + dualu(q0)
          do klong = ptr(j), ptr(j+1)-1
            i = row(klong)
            if (l(i) .ge. up) cycle
            ! dnew is new cost
            dnew = vj + val(klong)-dualu(i)
            ! Do not update d(i) if dnew ge cost of shortest path
            if (dnew .ge. csp) cycle
            if (iperm(i) .eq. 0) then
               ! Row i is unmatched; update shortest path info
               csp = dnew
               isp = klong
               jsp = j
            else
               ! Row i is matched; do not update d(i) if dnew is larger
               di = d(i)
               if (di .le. dnew) cycle
               if (l(i) .ge. low) cycle
               d(i) = dnew
               if (dnew .le. dmin) then
                  lpos = l(i)
                  if (lpos .ne. 0) call heap_delete(lpos,qlen,m,Q,D,L)
                  low = low - 1
                  q(low) = i
                  l(i) = low
               else
                  if (l(i) .eq. 0) then
                     qlen = qlen + 1
                     l(i) = qlen
                  end if
                  call heap_update(i,m,Q,D,L) ! d(i) has changed
               end if
               ! Update tree
               jj = iperm(i)
               out(jj) = klong
               pr(jj) = j
            end if
         end do
      end do

      ! If csp = RINF, no augmenting path is found
      if (csp .eq. RINF) GO TO 190
      ! Find augmenting path by tracing backward in pr; update iperm,jperm
      num = num + 1
      i = row(isp)
      iperm(i) = jsp
      jperm(jsp) = isp
      j = jsp
      do jdum = 1, num
         jj = pr(j)
         if (jj .eq. -1) exit
         klong = out(j)
         i = row(klong)
         iperm(i) = jj
         jperm(jj) = klong
         j = jj
      end do

      ! Update U for rows in q(up:m)
      do kk = up,m
         i = q(kk)
         dualu(i) = dualu(i) + d(i) - csp
      end do
190   do kk = low,m
         i = q(kk)
         d(i) = RINF
         l(i) = 0
      end do
      do kk = 1,qlen
         i = q(kk)
         d(i) = RINF
         l(i) = 0
      end do

   end do ! End of main loop


1000 continue
   ! Set dual column variables
   do j = 1, n
      klong = jperm(j)
      if (klong .ne. 0) then
         dualv(j) = val(klong) - dualu(row(klong))
      else
         dualv(j) = 0.0
      end if
   end do
   ! Zero dual row variables for unmatched rows
   where (iperm(1:m) .eq. 0) dualu(1:m) = 0.0

   ! Return if matrix has full structural rank
   if (num .eq. min(m,n)) return

   ! Otherwise, matrix is structurally singular, complete iperm.
   ! jperm, out are work arrays (now commented out)
!!$   jperm(1:n) = 0
!!$   k = 0
!!$   do i = 1, m
!!$      if (iperm(i) .eq. 0) then
!!$         k = k + 1
!!$         out(k) = i
!!$      else
!!$         j = iperm(i)
!!$         jperm(j) = i
!!$      end if
!!$   end do
!!$   k = 0
!!$   do j = 1, n
!!$      if (jperm(j) .ne. 0) cycle
!!$      k = k + 1
!!$      jdum = int(out(k))
!!$      iperm(jdum) = -j
!!$   end do
 end subroutine hungarian_match

!**********************************************************************
!
! Value associated with index i has decreased, update position in heap
! as approriate.
!
! This code is adapted from MC64 v 1.6.0
!
 subroutine heap_update(idx,N,Q,val,L)
   implicit none
   integer(ip_), intent(in) :: idx
   integer(ip_), intent(in) :: N
   integer(ip_), intent(inout) :: Q(N)
   integer(ip_), intent(inout) :: L(N)
   real(rp_), intent(in) :: val(N)

   integer(ip_) :: pos,parent_pos
   integer(ip_) :: parent_idx
   real(rp_) :: v

   ! Get current position of i in heap
   pos = L(idx)
   if (pos .le. 1) then
      ! idx is already at root of heap, but set q as it may have only just
      ! been inserted.
      q(pos) = idx
      return
   end if

   ! Keep trying to move i towards root of heap until it can't go any further
   v = val(idx)
   do while (pos .gt. 1) ! while not at root of heap
      parent_pos = pos / 2
      parent_idx = Q(parent_pos)
      ! If parent is better than idx, stop moving
      if (v .ge. val(parent_idx)) exit
      ! Otherwise, swap idx and parent
      Q(pos) = parent_idx
      L(parent_idx) = pos
      pos = parent_pos
   end do
   ! Finally set idx in the place it reached.
   Q(pos) = idx
   L(idx) = pos
 end subroutine heap_update

!**********************************************************************
!
! The root node is deleted from the binary heap.
!
! This code is adapted from MC64 v 1.6.0
!
 integer(ip_) function heap_pop(QLEN,N,Q,val,L)
   implicit none
   integer(ip_), intent(inout) :: QLEN
   integer(ip_), intent(in) :: N
   integer(ip_), intent(inout) :: Q(N)
   integer(ip_), intent(inout) :: L(N)
   real(rp_), intent(in) :: val(N)

   ! Return value is the old root of the heap
   heap_pop = q(1)

   ! Delete the root
   call heap_delete(1_ip_,QLEN,N,Q,val,L)
 end function heap_pop

!**********************************************************************
!
! Delete element in poisition pos0 from the heap
!
! This code is adapted from MC64 v 1.6.0
!
 subroutine heap_delete(pos0,QLEN,N,Q,D,L)
   implicit none
   integer(ip_) :: pos0,QLEN,N
   integer(ip_) :: Q(N),L(N)
   real(rp_) :: D(N)

   integer(ip_) :: idx,pos,parent,child,QK
   real(rp_) :: DK,DR,v

   ! If we're trying to remove the last item, just delete it.
   if (QLEN .eq. pos0) then
      QLEN = QLEN - 1
      return
   end if

   ! Replace index in position pos0 with last item and fix heap property
   idx = Q(QLEN)
   v = D(idx)
   QLEN = QLEN - 1 ! shrink heap
   pos = pos0 ! pos is current position of node I in the tree

   ! Move up if appropriate
   if (pos .gt. 1) then
      do
         parent = pos / 2
         QK = Q(parent)
         if (v .ge. D(QK)) exit
         Q(pos) = QK
         L(QK) = pos
         pos = parent
         if (pos .le. 1) exit
      end do
   end if
   Q(pos) = idx
   L(idx) = pos
   if (pos .ne. pos0) return ! Item moved up, hence doesn't need to move down

   ! Otherwise, move item down
   do
      child = 2 * pos
      if (child .gt. QLEN) exit
      DK = D(Q(child))
      if (child .lt. QLEN) then
         DR = D(Q(child+1))
         if (DK .gt. DR) then
            child = child + 1
            DK = DR
         end if
      end if
      if (v .le. DK) exit
      QK = Q(child)
      Q(pos) = QK
      L(QK) = pos
      pos = child
   end do
   Q(pos) = idx
   L(idx) = pos
 end subroutine heap_delete

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Auction Algorithm implementation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!
! An implementation of the auction algorithm to solve the assignment problem
! i.e. max_M sum_{(i,j)\in M} a_{ij}    where M is a matching.
! The dual variables u_i for row i and v_j for col j can be used to find
! a good scaling after postprocessing.
! We're aiming for:
! a_ij - u_i - v_j == 0    if (i,j) in M
! a_ij - u_i - v_j <= 0    otherwise
!
! Motivation of algorithm:
! Each unmatched column bids for its preferred row. Best bid wins.
! Prices (dual variables) are updated to reflect cost of using 2nd best instead
! for future auction rounds.
! i.e. Price of using entry (i,j) is a_ij - u_i
!
! In this implementation, only one column is bidding in each phase. This is
! largely equivalent to the motivation above but allows for faster progression
! as the same row can move through multiple partners during a single pass
! through the matrix.
!
 subroutine auction_match_core(m, n, ptr, row, val, match, dualu, dualv, &
      options, inform)
   implicit none
   integer(ip_), intent(in) :: m
   integer(ip_), intent(in) :: n
   integer(long_), dimension(n+1), intent(in) :: ptr
   integer(ip_), dimension(ptr(n+1)-1), intent(in) :: row
   real(rp_), dimension(ptr(n+1)-1), intent(in) :: val
   integer(ip_), dimension(n), intent(out) :: match
      ! match(j) = i => column j matched to row i
   real(rp_), dimension(m), intent(out) :: dualu ! row dual variables
   real(rp_), dimension(n), intent(inout) :: dualv ! col dual variables
   type(auction_options), intent(in) :: options
   type(auction_inform), intent(inout) :: inform

   integer(ip_), dimension(:), allocatable :: owner ! Inverse of match
   ! The list next(1:tail) is the search space of unmatched columns
   ! this is overwritten as we proceed such that next(1:insert) is the
   ! search space for the subsequent iteration.
   integer(ip_) :: tail, insert
   integer(ip_), dimension(:), allocatable :: next
   integer(ip_) :: unmatched ! Current number of unmatched cols

   integer(ip_) :: itr, minmn
   integer(ip_) :: i,  k
   integer(long_) :: j
   integer(ip_) :: col, cptr, bestr
   real :: ratio
   real(rp_) :: u, bestu, bestv

   real(rp_) :: eps ! minimum improvement

   integer(ip_) :: prev ! number of unmatched cols on previous iteration
   integer(ip_) :: nunchanged ! number of iterations where #unmatched cols has
      ! been constant

   inform%flag = 0
   inform%unmatchable = 0

   ! Allocate memory
   allocate(owner(m), next(n), stat=inform%stat)
   if (inform%stat .ne. 0) then
      inform%flag = ERROR_ALLOCATION
      return
   end if

   ! Set everything as unmatched
   minmn = min(m, n)
   unmatched = minmn
   match(1:n) = 0 ! 0 = unmatched, -1 = unmatched+ineligible
   owner(1:m) = 0
   dualu(1:m) = 0
   ! dualv is set for each column as it becomes matched, otherwise we use
   ! the value supplied on input (calculated as something sensible during
   ! preprocessing)

   ! Set up monitoring of progress
   prev = -1
   nunchanged = 0

   ! Initially all columns are unmatched
   tail = n
   do i = 1, n
      next(i) = i
   end do

   ! Iterate until we run out of unmatched buyers
   eps = options%eps_initial
   do itr = 1, options%max_iterations
      if (unmatched .eq. 0) exit ! nothing left to match
      ! Bookkeeping to determine number of iterations with no change
      if (unmatched .ne. prev) nunchanged = 0
      prev = unmatched
      nunchanged = nunchanged + 1
      ! Test if we satisfy termination conditions
      ratio = real(minmn-unmatched)/real(minmn)
      if ((nunchanged                  .ge. options%max_unchanged(1)) .and. &
         (ratio .ge. options%min_proportion(1))) exit
      if ((nunchanged                  .ge. options%max_unchanged(2)) .and. &
         (ratio .ge. options%min_proportion(2))) exit
      if ((nunchanged                  .ge. options%max_unchanged(3)) .and. &
         (ratio .ge. options%min_proportion(3))) exit
      ! Update epsilon scaling
      eps = min(1.0_rp_, eps+1.0_rp_/real(n+1,rp_))
      ! Now iterate over all unmatched entries listed in next(1:tail)
      ! As we progress, build list for next iteration in next(1:insert)
      insert = 0
      do cptr = 1, tail
         col = next(cptr)
         if (match(col) .ne. 0) cycle ! already matched or ineligible
         if (ptr(col) .eq. ptr(col+1)) cycle ! empty col (only ever fails on
           ! first iteration - not put in next(1:insert) thereafter)
         ! Find best value of a_ij - u_i for current column
         ! This occurs for i=bestr with value bestu
         ! second best value stored as bestv
         j = ptr(col)
         bestr = row(j)
         bestu = val(j) - dualu(bestr)
         bestv = -huge(bestv)
         do j = ptr(col)+1, ptr(col+1)-1
            u = val(j) - dualu(row(j))
            if (u .gt. bestu) then
               bestv = bestu
               bestr = row(j)
               bestu = u
            else if (u .gt. bestv) then
               bestv = u
            end if
         end do
         if (bestv .eq. -huge(bestv)) bestv = 0.0 ! No second best
         ! Check if matching this column gives us a net benefit
         if (bestu .gt. 0) then
            ! There is a net benefit, match column col to row bestr
            ! if bestr was previously matched to col k, unmatch it
            dualu(bestr) = dualu(bestr) + bestu - bestv + eps
            dualv(col) = bestv - eps ! satisfy a_ij - u_i - v_j = 0
            match(col) = bestr
            unmatched = unmatched - 1
            k = owner(bestr)
            owner(bestr) = col
            if (k .ne. 0) then
               ! Mark column k as unmatched
               match(k) = 0 ! unmatched
               unmatched = unmatched + 1
               insert = insert + 1
               next(insert) = k
            end if
         else
            ! No net benefit, mark col as ineligible for future consideration
            match(col) = -1 ! ineligible
            unmatched = unmatched - 1
            inform%unmatchable = inform%unmatchable + 1
         end if
      end do
      tail = insert
   end do
   inform%iterations = itr-1

   ! We expect unmatched columns to have match(col) = 0
   where(match(:) .eq. -1) match(:) = 0
 end subroutine auction_match_core

! Find a scaling through a matching-based approach using the auction algorithm
! This subroutine actually performs pre/post-processing around the call to
! auction_match_core() to actually use the auction algorithm
!
! This consists of finding a2_ij = 2*maxentry - cmax + log(|a_ij|)
! where cmax is the log of the absolute maximum in the column
! and maxentry is the maximum value of cmax-log(|a_ij|) across entire matrix
! The cmax-log(|a_ij|) term converts from max product to max sum problem and
! normalises scaling across matrix. The 2*maxentry term biases the result
! towards a high cardinality solution.
!
! Lower triangle only as input (log(half)+half->full faster than log(full))
!
 subroutine auction_match(expand, m, n, ptr, row, val, match, rscaling, &
      cscaling, options, inform)
   implicit none
   logical, intent(in) :: expand
   integer(ip_), intent(in) :: m
   integer(ip_), intent(in) :: n
   integer(long_), dimension(n+1), intent(in) :: ptr
   integer(ip_), dimension(*), intent(in) :: row
   real(rp_), dimension(*), intent(in) :: val
   integer(ip_), dimension(m), intent(out) :: match
   real(rp_), dimension(m), intent(out) :: rscaling
   real(rp_), dimension(n), intent(out) :: cscaling
   type(auction_options), intent(in) :: options
   type(auction_inform), intent(inout) :: inform

   integer(long_), allocatable :: ptr2(:)
   integer(ip_), allocatable :: row2(:), iw(:), cmatch(:)
   real(rp_), allocatable :: val2(:), cmax(:)
   real(rp_) :: colmax
   integer(ip_) :: i
   integer(long_) :: jlong, klong
   integer(long_) :: ne
   real(rp_), parameter :: zero = 0.0
   real(rp_) :: maxentry

   inform%flag = 0

   ! Reset ne for the expanded symmetric matrix
   ne = ptr(n+1)-1
   ne = 2*ne

   ! Expand matrix, drop explicit zeroes and take log absolute values
   allocate(ptr2(n+1), row2(ne), val2(ne), cmax(n), cmatch(n), &
      stat=inform%stat)
   if (inform%stat .ne. 0) then
      inform%flag = ERROR_ALLOCATION
      return
   end if

   klong = 1
   do i = 1, n
      ptr2(i) = klong
      do jlong = ptr(i), ptr(i+1)-1
         if (val(jlong) .eq. zero) cycle
         row2(klong) = row(jlong)
         val2(klong) = abs(val(jlong))
         klong = klong + 1
      end do
      ! Following log is seperated from above loop to expose expensive
      ! log operation to vectorization.
      val2(ptr2(i):klong-1) = log(val2(ptr2(i):klong-1))
   end do
   ptr2(n+1) = klong
   if (expand) then
      if (m .ne. n) then
         ! Should never get this far with a non-square matrix
         inform%flag = -99
         return
      end if
      allocate(iw(5*n), stat=inform%stat)
      if (inform%stat .ne. 0) then
         inform%flag = ERROR_ALLOCATION
         return
      end if
      call half_to_full(n, row2, ptr2, iw, a=val2)
   end if

   ! Compute column maximums
   do i = 1, n
      if (ptr2(i+1) .le. ptr2(i)) then
         ! Empty col
         cmax(i) = 0.0
         cycle
      end if
      colmax = maxval(val2(ptr2(i):ptr2(i+1)-1))
      cmax(i) = colmax
      val2(ptr2(i):ptr2(i+1)-1) = colmax - val2(ptr2(i):ptr2(i+1)-1)
   end do

   maxentry = maxval(val2(1:ptr2(n+1)-1))
   ! Use 2*maxentry+1 to prefer high cardinality matchings (+1 avoids 0 cols)
   maxentry = 2*maxentry+1
   val2(1:ptr2(n+1)-1) = maxentry - val2(1:ptr2(n+1)-1)
   !cscaling(1:n) = maxentry - cmax(1:n) ! equivalent to scale=1.0 for unmatched
   !   ! cols that core algorithm doesn't visit
   cscaling(1:n) = - cmax(1:n) ! equivalent to scale=1.0 for unmatched
     ! cols that core algorithm doesn't visit

   call auction_match_core(m, n, ptr2, row2, val2, cmatch, rscaling, &
        cscaling, options, inform)
   inform%matched = count(cmatch.ne.0)

   ! Calculate an adjustment so row and col scaling similar orders of magnitude
   ! and undo pre processing
   rscaling(1:m) = -rscaling(1:m) + maxentry
   cscaling(1:n) = -cscaling(1:n) - cmax(1:n)

   ! Convert row->col matching into col->row one
   match(1:m) = 0
   do i = 1, n
      if (cmatch(i) .eq. 0) cycle ! unmatched row
      match(cmatch(i)) = i
   end do
   call match_postproc(m, n, ptr, row, val, rscaling, cscaling, &
        inform%matched, match, inform%flag, inform%stat)
 end subroutine auction_match

 subroutine match_postproc(m, n, ptr, row, val, rscaling, cscaling, nmatch, &
      match, flag, st)
   implicit none
   integer(ip_), intent(in) :: m
   integer(ip_), intent(in) :: n
   integer(long_), dimension(n+1), intent(in) :: ptr
   integer(ip_), dimension(ptr(n+1)-1), intent(in) :: row
   real(rp_), dimension(ptr(n+1)-1), intent(in) :: val
   real(rp_), dimension(m), intent(inout) :: rscaling
   real(rp_), dimension(n), intent(inout) :: cscaling
   integer(ip_), intent(in) :: nmatch
   integer(ip_), dimension(m), intent(in) :: match
   integer(ip_), intent(inout) :: flag
   integer(ip_), intent(inout) :: st

   integer(ip_) :: i
   integer(long_) :: jlong
   real(rp_), dimension(:), allocatable :: rmax, cmax
   real(rp_) :: ravg, cavg, adjust, colmax, v

   if (m .eq. n) then
      ! Square
      ! Just perform post-processing and magnitude adjustment
      ravg = sum(rscaling(1:m)) / real(m,rp_)
      cavg = sum(cscaling(1:n)) / real(n,rp_)
      adjust = (ravg - cavg) / 2
      rscaling(1:m) = rscaling(1:m) - adjust
      cscaling(1:n) = cscaling(1:n) + adjust
   else if (m .lt. n) then
      ! More columns than rows
      ! Allocate some workspace
      allocate(cmax(n), stat=st)
      if (st .ne. 0) then
         flag = ERROR_ALLOCATION
         return
      end if
      ! First perform post-processing and magnitude adjustment based on match
      ravg = 0
      cavg = 0
      do i = 1, m
         if (match(i) .eq. 0) cycle
         ravg = ravg + rscaling(i)
         cavg = cavg + cscaling(match(i))
      end do
      ravg = ravg / real(nmatch,rp_)
      cavg = cavg / real(nmatch,rp_)
      adjust = (ravg - cavg) / 2
      rscaling(1:m) = rscaling(1:m) - adjust
      cscaling(1:n) = cscaling(1:n) + adjust
      ! For each unmatched col, scale max entry to 1.0
      do i = 1, n
         colmax = 0.0
         do jlong = ptr(i), ptr(i+1)-1
            v = abs(val(jlong)) * exp( rscaling(row(jlong)) )
            colmax = max(colmax, v)
         end do
         if (colmax .eq. 0.0) then
            cmax(i) = 0.0
         else
            cmax(i) = log(1/colmax)
         end if
      end do
      do i = 1, m
         if (match(i) .eq. 0) cycle
         cmax(match(i)) = cscaling(match(i))
      end do
      cscaling(1:n) = cmax(1:n)
   else if (n .lt. m) then
      ! More rows than columns
      ! Allocate some workspace
      allocate(rmax(m), stat=st)
      if (st .ne. 0) then
         flag = ERROR_ALLOCATION
         return
      end if
      ! First perform post-processing and magnitude adjustment based on match
      ! also record which rows have been matched
      ravg = 0
      cavg = 0
      do i = 1, m
         if (match(i) .eq. 0) cycle
         ravg = ravg + rscaling(i)
         cavg = cavg + cscaling(match(i))
      end do
      ravg = ravg / real(nmatch,rp_)
      cavg = cavg / real(nmatch,rp_)
      adjust = (ravg - cavg) / 2
      rscaling(1:m) = rscaling(1:m) - adjust
      cscaling(1:n) = cscaling(1:n) + adjust
      ! Find max column-scaled value in each row from unmatched cols
      rmax(:) = 0.0
      do i = 1, n
         do jlong = ptr(i), ptr(i+1)-1
            v = abs(val(jlong)) * exp( cscaling(i) )
            rmax(row(jlong)) = max(rmax(row(jlong)), v)
         end do
      end do
      ! Calculate scaling for each row, but overwrite with correct values for
      ! matched rows, then copy entire array over rscaling(:)
      do i = 1, m
         if (match(i) .ne. 0) cycle
         if (rmax(i) .eq. 0.0) then
            rscaling(i) = 0.0
         else
            rscaling(i) =  log( 1 / rmax(i) )
         end if
      end do
   end if
 end subroutine match_postproc

end module spral_scaling_precision
