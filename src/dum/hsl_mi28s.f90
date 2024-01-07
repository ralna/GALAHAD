! THIS VERSION: GALAHAD 4.3 - 2024-01-06 AT 10:15 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M I 2 8   M O D U L E  -*-*-*-

 module hsl_mi28_single
   USE GALAHAD_SYMBOLS
   USE GALAHAD_KINDS
   implicit none

   private
   public :: mi28_keep, mi28_control, mi28_info
   public :: mi28_factorize, mi28_finalise, mi28_precondition, mi28_solve

  real(sp_), parameter :: zero = 0.0_sp_
  real(sp_), parameter :: one = 1.0_sp_
  real(sp_), parameter :: sfact = 2.0_sp_
  real(sp_), parameter :: sfact2 = 4.0_sp_
  real(sp_), parameter :: alpham = 0.001_sp_

  interface mi28_factorize
      module procedure mi28_factorize_single
  end interface

  interface mi28_precondition
      module procedure mi28_precondition_single
  end interface

  interface mi28_solve
      module procedure mi28_solve_single
  end interface

  interface mi28_finalise
      module procedure mi28_finalise_single
  end interface

  type mi28_keep
    integer(long_), allocatable ::  fact_ptr(:)
    integer(ip_),  allocatable ::  fact_row(:) 
    real(sp_), allocatable ::  fact_val(:) 
    real(sp_), allocatable :: scale(:) 
    integer(ip_),  allocatable :: invp(:)
    integer(ip_),  allocatable :: perm(:)
    real(sp_), allocatable :: w(:)
  end type mi28_keep

  type mi28_control
    real(sp_) :: alpha = zero
    logical :: check = .true.
    integer(ip_) :: iorder = 6
    integer(ip_) :: iscale = 1
    real(sp_) :: lowalpha = alpham
    integer(ip_) :: maxshift = 3
    logical :: rrt = .false.
    real(sp_) :: shift_factor = sfact
    real(sp_) :: shift_factor2 = sfact2
    real(sp_) :: small = 10.0_sp_**(-20)
    real(sp_) :: tau1 = 0.001_sp_
    real(sp_) :: tau2 = 0.0001_sp_
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
    real(sp_) :: profile_before = 0 
    real(sp_) :: profile_after = 0 
    integer(long_) :: size_r = 0_long_ 
    integer(ip_) :: stat = 0 
    real(sp_) :: alpha = zero 
  end type mi28_info

 contains

  subroutine mi28_factorize_single(n, ptr, row, val, lsize, rsize, keep,       &
      control, info, scale, invp)
    integer(ip_),  intent(in) :: n  
    integer(ip_),  intent(inout) ::  ptr(n+1) 
    integer(ip_),  intent(inout) ::  row(:) 
    real(sp_), intent(inout) ::  val(:) 
    integer(ip_),  intent(in) :: lsize 
    integer(ip_),  intent(in) :: rsize 
    type(mi28_keep), intent(out) :: keep 
    type(mi28_control), intent(in) :: control 
    type(mi28_info), intent(out) :: info      
    real(sp_), intent(in), optional :: scale(n) 
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
  end subroutine mi28_factorize_single

  subroutine mi28_precondition_single(n, keep, z, y, info)
    integer(ip_),  intent(in) :: n
    type(mi28_keep), intent(inout) :: keep
    real(sp_), intent(in) :: z(n)
    real(sp_), intent(out) :: y(n)
    type(mi28_info), intent(inout) :: info
    info%flag = GALAHAD_unavailable_option
  end subroutine mi28_precondition_single

  subroutine mi28_solve_single(trans, n, keep, z, y, info)
    logical, intent(in) :: trans 
    integer(ip_),  intent(in) :: n
    type(mi28_keep), intent(inout) :: keep
    real(sp_), intent(in) :: z(n)
    real(sp_), intent(out) :: y(n)
    type(mi28_info), intent(inout) :: info
    info%flag = GALAHAD_unavailable_option

  end subroutine mi28_solve_single

  subroutine mi28_finalise_single(keep, info)
    type(mi28_keep), intent(inout) :: keep
    type(mi28_info), intent(inout) :: info
    info%flag = GALAHAD_unavailable_option
  end subroutine mi28_finalise_single

 end module hsl_mi28_single
     
