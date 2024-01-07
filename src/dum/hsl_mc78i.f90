! THIS VERSION: GALAHAD 4.3 - 2024-01-06 AT 08:30 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M C 7 8    M O D U L E  -*-*-*-

module hsl_mc78_integer
   USE GALAHAD_KINDS
   implicit none

   private
   public :: mc78_control
   public :: mc78_analyse

   type mc78_control
      integer(ip_) :: heuristic = 1 ! 1=ma77 2=cholmod
      integer(ip_) :: nrelax(3) = (/ 4, 16, 48 /) ! CHOLMOD-like
      real(dp_) :: zrelax(3) = (/ 0.8, 0.1, 0.05 /) ! CHOLMOD-like
      integer(ip_) :: nemin = 16  ! Node amalgamation parameter

      integer(ip_) :: unit_error = 6
      integer(ip_) :: unit_warning = 6
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

   integer(ip_), parameter :: MC78_ERROR_ALLOC = -1
   integer(ip_), parameter :: MC78_ERROR_SSA   = -2
   integer(ip_), parameter :: MC78_ERROR_ROW_SMALL = -3
   integer(ip_), parameter :: MC78_ERROR_UNKNOWN = -99

   ! Warning flags are treated as bit masks, add together if multiple occour
   integer(ip_), parameter :: MC78_WARNING_SSA = 1
   integer(ip_), parameter :: MC78_WARNING_BLK_SVAR = 2

   interface mc78_analyse
      module procedure mc78_analyse_assembled_integer
      module procedure mc78_analyse_elemental_integer
   end interface mc78_analyse

contains

subroutine mc78_analyse_assembled_integer(n, ptr, row, perm, nnodes, sptr, &
      sparent, rptr, rlist, control, info, stat, nfact, nflops, piv_size)
   USE GALAHAD_SYMBOLS
   integer(ip_), intent(in) :: n ! Dimension of system
   integer(ip_), dimension(n+1), intent(in) :: ptr ! Col pointers
   integer(ip_), dimension(ptr(n+1)-1), intent(in) :: row ! Row indices
   integer(ip_), dimension(n), intent(inout) :: perm
      ! perm(i) must hold position of i in the pivot sequence. 
      ! On exit, holds the pivot order to be used by factorization.
   integer(ip_) :: nnodes ! number of supernodes found
   integer(ip_), dimension(:), allocatable :: sptr ! supernode pointers
   integer(ip_), dimension(:), allocatable :: sparent ! assembly tree
   integer(long_), dimension(:), allocatable :: rptr
      ! pointers to rlist
   integer(ip_), dimension(:), allocatable :: rlist ! row lists
   ! For details of control, info : see derived type descriptions
   type(mc78_control), intent(in) :: control
   integer(ip_) :: info
   integer(ip_), optional :: stat
   integer(long_), optional :: nfact
   integer(long_), optional :: nflops
   integer(ip_), dimension(n), optional, intent(inout) :: piv_size ! If
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
   integer(ip_), intent(in) :: n ! Maximum integer used to index an element
   integer(ip_), intent(in) :: nelt ! Number of elements
   integer(ip_), dimension(nelt+1), intent(in) :: starts ! Element pointers
   integer(ip_), dimension(starts(nelt+1)-1), intent(in) :: vars ! Variables
      !assoicated with each element. Element i has vars(starts(i):starts(i+1)-1)
   integer(ip_), dimension(n), intent(inout) :: perm
      ! perm(i) must hold position of i in the pivot sequence. 
      ! On exit, holds the pivot order to be used by factorization.
   integer(ip_), dimension(nelt) :: eparent ! On exit, eparent(i) holds
      ! node of assembly that element i is a child of.
   integer(ip_) :: nnodes ! number of supernodes found
   integer(ip_), dimension(:), allocatable :: sptr ! supernode pointers
   integer(ip_), dimension(:), allocatable :: sparent ! assembly tree
   integer(long_), dimension(:), allocatable :: rptr
      ! pointers to rlist
   integer(ip_), dimension(:), allocatable :: rlist ! row lists
   ! For details of control, info : see derived type descriptions
   type(mc78_control), intent(in) :: control
   integer(ip_) :: info
   integer(ip_), optional :: stat
   integer(long_), optional :: nfact ! If present, then on exit
      ! contains the number of entries in L
   integer(long_), optional :: nflops ! If present, then on exit
      ! contains the number of floating point operations in factorize.
   integer(ip_), dimension(n), optional, intent(inout) :: piv_size ! If
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
