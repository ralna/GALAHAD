! THIS VERSION: 25/10/2013 AT 14:00:00 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M I 2 8   M O D U L E  -*-*-*-

 module hsl_mi28_double
   USE GALAHAD_SYMBOLS
   implicit none

   private
   public :: mi28_keep, mi28_control, mi28_info
   public :: mi28_factorize, mi28_finalise, mi28_precondition, mi28_solve

  integer, parameter  :: wp = kind(0.0d0)
  integer, parameter :: long = selected_int_kind(18)
  real(wp), parameter :: zero = 0.0_wp
  real(wp), parameter :: one = 1.0_wp
  real(wp), parameter :: sfact = 2.0_wp
  real(wp), parameter :: sfact2 = 4.0_wp
  real(wp), parameter :: alpham = 0.001_wp

  interface mi28_factorize
      module procedure mi28_factorize_double
  end interface

  interface mi28_precondition
      module procedure mi28_precondition_double
  end interface

  interface mi28_solve
      module procedure mi28_solve_double
  end interface

  interface mi28_finalise
      module procedure mi28_finalise_double
  end interface

  type mi28_keep
    integer(long), allocatable ::  fact_ptr(:)
    integer, allocatable ::  fact_row(:) 
    real(wp), allocatable ::  fact_val(:) 
    real(wp), allocatable :: scale(:) 
    integer, allocatable :: invp(:)
    integer, allocatable :: perm(:)
    real(wp), allocatable :: w(:)
  end type mi28_keep

  type mi28_control
    real(wp) :: alpha = zero
    logical :: check = .true.
    integer :: iorder = 6
    integer :: iscale = 1
    real(wp) :: lowalpha = alpham
    integer :: maxshift = 3
    logical :: rrt = .false.
    real(wp) :: shift_factor = sfact
    real(wp) :: shift_factor2 = sfact2
    real(wp) :: small = 10.0_wp**(-20)
    real(wp) :: tau1 = 0.001_wp
    real(wp) :: tau2 = 0.0001_wp
    integer :: unit_error = 6
    integer :: unit_warning = 6
  end type mi28_control

  type mi28_info
    integer :: band_after = 0 
    integer :: band_before = 0 
    integer :: dup = 0 
    integer :: flag = 0 
    integer :: flag61 = 0 
    integer :: flag64 = 0 
    integer :: flag68 = 0 
    integer :: flag77 = 0 
    integer :: nrestart = 0 
    integer :: nshift = 0 
    integer :: oor = 0 
    real(wp) :: profile_before = 0 
    real(wp) :: profile_after = 0 
    integer(long) :: size_r = 0_long 
    integer :: stat = 0 
    real(wp) :: alpha = zero 
  end type mi28_info

 contains

  subroutine mi28_factorize_double(n, ptr, row, val, lsize, rsize, keep,       &
      control, info, scale, invp)
    integer, intent(in) :: n  
    integer, intent(inout) ::  ptr(n+1) 
    integer, intent(inout) ::  row(:) 
    real(wp), intent(inout) ::  val(:) 
    integer, intent(in) :: lsize 
    integer, intent(in) :: rsize 
    type(mi28_keep), intent(out) :: keep 
    type(mi28_control), intent(in) :: control 
    type(mi28_info), intent(out) :: info      
    real(wp), intent(in), optional :: scale(n) 
    integer, intent(in), optional :: invp(n) 
    IF ( control%unit_error >= 0 ) WRITE( control%unit_error,                  &
           "( ' We regret that the solution options that you have ', /,        &
   &     ' chosen are not all freely available with GALAHAD.', /,              &
   &     ' If you have HSL (formerly the Harwell Subroutine', /,               &
   &     ' Library), this option may be enabled by replacing the dummy ', /,   &
   &     ' subroutine MI28_factorize with its HSL namesake ', /,               &
   &     ' and dependencies. See ', /,                                         &
   &     '   $GALAHAD/src/makedefs/packages for details.' )" )                
    info%flag = GALAHAD_unavailable_option
  end subroutine mi28_factorize_double

  subroutine mi28_precondition_double(n, keep, z, y, info)
    integer, intent(in) :: n
    type(mi28_keep), intent(inout) :: keep
    real(wp), intent(in) :: z(n)
    real(wp), intent(out) :: y(n)
    type(mi28_info), intent(inout) :: info
    info%flag = GALAHAD_unavailable_option
  end subroutine mi28_precondition_double

  subroutine mi28_solve_double(trans, n, keep, z, y, info)
    logical, intent(in) :: trans 
    integer, intent(in) :: n
    type(mi28_keep), intent(inout) :: keep
    real(wp), intent(in) :: z(n)
    real(wp), intent(out) :: y(n)
    type(mi28_info), intent(inout) :: info
    info%flag = GALAHAD_unavailable_option

  end subroutine mi28_solve_double

  subroutine mi28_finalise_double(keep, info)
    type(mi28_keep), intent(inout) :: keep
    type(mi28_info), intent(inout) :: info
    info%flag = GALAHAD_unavailable_option
  end subroutine mi28_finalise_double

 end module hsl_mi28_double
     
