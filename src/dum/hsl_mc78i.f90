! THIS VERSION: 20/01/2011 AT 12:30:00 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M C 7 8    M O D U L E  -*-*-*-

module hsl_mc78_integer
   implicit none

   private
   public :: mc78_control
   public :: mc78_analyse

   integer, parameter :: dp = kind(0d0) ! not package type
   integer, parameter :: long = selected_int_kind(18)

   integer, parameter :: pkg_type = kind(0) ! package type - integer or long

   type mc78_control
      integer :: heuristic = 1 ! 1=ma77 2=cholmod
      integer :: nrelax(3) = (/ 4, 16, 48 /) ! CHOLMOD-like
      real(dp) :: zrelax(3) = (/ 0.8, 0.1, 0.05 /) ! CHOLMOD-like
      integer :: nemin = 16  ! Node amalgamation parameter

      integer :: unit_error = 6
      integer :: unit_warning = 6
      logical :: ssa_abort = .false. ! If .true., then return with an error if
         ! an assembled matrix is detected as symbolically singular (we do
         ! not garuntee to detect _all_ symbolically singular matrices).
         ! If .false., then a warning is raised instead.
      logical :: svar = .false. ! If .true. then supervariables are used in
         ! the assembled case, otherwise they are not. Supervaraibles are
         ! always used in the elemental case.
      logical :: sort = .false. ! If .true. then entries within each supernode's
         ! row lists are sorted. Otherwise they might not be.
      logical :: lopt = .false. ! If .true. then variable ordering is optimized
         ! for cache locality. Otherwise it is not.
   end type mc78_control

   integer, parameter :: MC78_ERROR_ALLOC = -1 ! allocate error
   integer, parameter :: MC78_ERROR_SSA   = -2 ! symbolically singular assembled
   integer, parameter :: MC78_ERROR_ROW_SMALL = -3 ! supplied row array to short
   integer, parameter :: MC78_ERROR_UNKNOWN = -99 ! internal/unknown error

   ! Warning flags are treated as bit masks, add together if multiple occour
   integer, parameter :: MC78_WARNING_SSA = 1 ! symbolically singular assembled
   integer, parameter :: MC78_WARNING_BLK_SVAR = 2 ! svar and blk pivs requested

   interface mc78_analyse
      module procedure mc78_analyse_assembled_integer
      module procedure mc78_analyse_elemental_integer
   end interface mc78_analyse

contains

subroutine mc78_analyse_assembled_integer(n, ptr, row, perm, nnodes, sptr, &
      sparent, rptr, rlist, control, info, stat, nfact, nflops, piv_size)
   USE GALAHAD_SYMBOLS
   integer, intent(in) :: n ! Dimension of system
   integer(pkg_type), dimension(n+1), intent(in) :: ptr ! Column pointers
   integer, dimension(ptr(n+1)-1), intent(in) :: row ! Row indices
   integer, dimension(n), intent(inout) :: perm
      ! perm(i) must hold position of i in the pivot sequence. 
      ! On exit, holds the pivot order to be used by factorization.
   integer :: nnodes ! number of supernodes found
   integer, dimension(:), allocatable :: sptr ! supernode pointers
   integer, dimension(:), allocatable :: sparent ! assembly tree
   integer(long), dimension(:), allocatable :: rptr
      ! pointers to rlist
   integer, dimension(:), allocatable :: rlist ! row lists
   ! For details of control, info : see derived type descriptions
   type(mc78_control), intent(in) :: control
   integer :: info
   integer, optional :: stat
   integer(long), optional :: nfact
   integer(long), optional :: nflops
   integer, dimension(n), optional, intent(inout) :: piv_size ! If
      ! present, then matches matrix order and specifies block pivots. 
      ! piv_size(i) is the number of entries pivots in the block pivot
      ! containing column i.

        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MC78_analyse_assembled with its HSL namesake ', /,        &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

   info = GALAHAD_unavailable_option

end subroutine mc78_analyse_assembled_integer

subroutine mc78_analyse_elemental_integer(n, nelt, starts, vars, perm, &
      eparent, nnodes, sptr, sparent, rptr, rlist, control, info, stat, &
      nfact, nflops, piv_size)
   USE GALAHAD_SYMBOLS
   integer, intent(in) :: n ! Maximum integer used to index an element
   integer, intent(in) :: nelt ! Number of elements
   integer(pkg_type), dimension(nelt+1), intent(in) :: starts ! Element pointers
   integer, dimension(starts(nelt+1)-1), intent(in) :: vars ! Variables
      !assoicated with each element. Element i has vars(starts(i):starts(i+1)-1)
   integer, dimension(n), intent(inout) :: perm
      ! perm(i) must hold position of i in the pivot sequence. 
      ! On exit, holds the pivot order to be used by factorization.
   integer, dimension(nelt) :: eparent ! On exit, eparent(i) holds
      ! node of assembly that element i is a child of.
   integer :: nnodes ! number of supernodes found
   integer, dimension(:), allocatable :: sptr ! supernode pointers
   integer, dimension(:), allocatable :: sparent ! assembly tree
   integer(long), dimension(:), allocatable :: rptr
      ! pointers to rlist
   integer, dimension(:), allocatable :: rlist ! row lists
   ! For details of control, info : see derived type descriptions
   type(mc78_control), intent(in) :: control
   integer :: info
   integer, optional :: stat
   integer(long), optional :: nfact ! If present, then on exit
      ! contains the number of entries in L
   integer(long), optional :: nflops ! If present, then on exit
      ! contains the number of floating point operations in factorize.
   integer, dimension(n), optional, intent(inout) :: piv_size ! If
      ! present, then matches matrix order and specifies block pivots. 
      ! piv_size(i) is the number of entries pivots in the block pivot
      ! containing column i.

        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MC78_analyse_element with its HSL namesake ', /,          &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

   info = GALAHAD_unavailable_option

end subroutine mc78_analyse_elemental_integer

end module hsl_mc78_integer
