! THIS VERSION: 29/05/2015 AT 11:40:00 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M I 3 5   M O D U L E  -*-*-*-

  module hsl_mi35_single
    USE GALAHAD_SYMBOLS
    implicit none

    private
    public :: mi35_keep, mi35_control, mi35_info
    public :: mi35_factorize, mi35_finalise, mi35_precondition, mi35_solve
    public :: mi35_check_matrix, mi35_factorizeC, mi35_formC

    integer, parameter  :: wp = kind(0.0e0)
    integer, parameter  :: long = selected_int_kind(18)
    real(wp), parameter :: zero = 0.0_wp
    real(wp), parameter :: one = 1.0_wp
    real(wp), parameter :: sfact = 2.0_wp
    real(wp), parameter :: sfact2 = 4.0_wp
    real(wp), parameter :: alpham = 0.001_wp

    type mi35_keep
      integer(long), allocatable ::  fact_ptr(:)
      integer, allocatable ::  fact_row(:)
      real(wp), allocatable ::  fact_val(:)
      real(wp), allocatable :: scale(:)
      integer, allocatable :: invp(:)
      integer, allocatable :: perm(:)
      real(wp), allocatable :: w(:)
    end type mi35_keep

    type mi35_control
      real(wp) :: alpha = zero
      integer :: iorder = 6
      integer :: iscale = 1
      integer :: limit_rowA = -1
      integer :: limit_colC = -1
      integer :: limit_C = -1
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
    end type mi35_control

    type mi35_info
      real(wp) :: avlenC = zero
      integer :: band_after = 0
      integer :: band_before = 0
      integer :: dup = 0
      integer :: flag = 0
      integer :: flag61 = 0
      integer :: flag64 = 0
      integer :: flag68 = 0
      integer :: flag77 = 0
      integer :: maxlen = 0
      integer :: maxlenC = 0
      integer :: nrestart = 0
      integer :: nshift = 0
      integer :: nnz_C = 0
      integer :: nzero = 0
      integer :: nzero_weight = 0
      integer :: oor = 0
      real(wp) :: profile_before = 0
      real(wp) :: profile_after = 0
      integer(long) :: size_r = 0_long
      integer :: stat = 0
      real(wp) :: alpha = zero
    end type mi35_info

  contains

    subroutine mi35_check_matrix( m, n, ptr, row, val, control, info, weight, b)
    integer, intent(inout) :: m
    integer, intent(inout) :: n
    integer, intent(inout) ::  ptr(n+1)
    integer, intent(inout) ::  row(:)
    real(wp), intent(inout) ::  val(:)
    type(mi35_info), intent(out) :: info
    type(mi35_control), intent(in) :: control
    real(wp), intent(inout), optional :: weight(m)
    real(wp), intent(inout), optional :: b(m)
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
    integer, intent(in) :: m
    integer, intent(in) :: n
    integer, intent(in) ::  ptr(n+1)
    integer, intent(in) ::  row(:)
    real(wp), intent(in) ::  val(:)
    integer, intent(in) :: lsize
    integer, intent(in) :: rsize
    type(mi35_keep), intent(out) :: keep
    type(mi35_control), intent(in) :: control
    type(mi35_info), intent(out) :: info
    real(wp), intent(in), optional :: weight(m)
    real(wp), intent(in), optional :: scale(n)
    integer, intent(in), optional :: perm(n)
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

    subroutine mi35_formC( m, n, ptrA, rowA, valA, ptrC, rowC, valC,           &
                           control, info, weight)
    integer, intent(in) :: m
    integer, intent(in) :: n
    integer, intent(in) ::  ptrA(n+1)
    integer, intent(in) ::  rowA(:)
    real(wp), intent(in) ::  valA(:)
    integer, intent(out) ::  ptrC(n+1)
    integer, intent(out), allocatable ::  rowC(:)
    real(wp), intent(out), allocatable ::  valC(:)
    type(mi35_control), intent(in) :: control
    type(mi35_info), intent(inout) :: info
    real(wp), optional, intent(in) ::  weight(m)
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

    subroutine mi35_factorizeC(n, ptr, row, val, lsize, rsize, keep,           &
                               control, info, scale, perm)
    integer, intent(in) :: n
    integer, intent(in) ::  ptr(n+1)
    integer, intent(in) ::  row(:)
    real(wp), intent(in) ::  val(:)
    integer, intent(in) :: lsize
    integer, intent(in) :: rsize
    type(mi35_keep), intent(out) :: keep
    type(mi35_control), intent(in) :: control
    type(mi35_info), intent(inout) :: info
    real(wp), intent(in), optional :: scale(n)
    integer, intent(in), optional :: perm(n)
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
    integer, intent(in) :: n
    type(mi35_keep), intent(inout) :: keep
    real(wp), intent(in) :: z(n)
    real(wp), intent(out) :: y(n)
    type(mi35_info), intent(inout) :: info
    info%flag = GALAHAD_unavailable_option
    end subroutine mi35_precondition

!****************************************************************************

    subroutine mi35_solve( trans, n, keep, z, y, info )
    logical, intent(in) :: trans
    integer, intent(in) :: n
    type(mi35_keep), intent(inout) :: keep
    real(wp), intent(in) :: z(n)
    real(wp), intent(out) :: y(n)
    type(mi35_info), intent(inout) :: info
    info%flag = GALAHAD_unavailable_option
    end subroutine mi35_solve

!****************************************************************************

    subroutine mi35_finalise( keep, info )
    type(mi35_keep), intent(inout) :: keep
    type(mi35_info), intent(inout) :: info
    info%flag = GALAHAD_unavailable_option
    end subroutine mi35_finalise

  end module hsl_mi35_single
