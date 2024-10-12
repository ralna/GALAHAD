! THIS VERSION: GALAHAD 5.1 - 2024-10-11 AT 14:30 GMT.

#include "hsl_subset.h"

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M A 8 6    M O D U L E  -*-*-*-

module hsl_ma86_real
!$ use omp_lib
   use hsl_kinds_real, only: ip_, long_, lp_, rp_
   use hsl_mc78_integer
   use hsl_mc34_real
   use hsl_zd11_real
#ifdef INTEGER_64
   USE GALAHAD_SYMBOLS_64, ONLY: GALAHAD_unavailable_option
#else
   USE GALAHAD_SYMBOLS, ONLY: GALAHAD_unavailable_option
#endif

   implicit none
   public :: ma86_keep, ma86_control, ma86_info
   public :: ma86_analyse, ma86_factor, ma86_factor_solve, ma86_solve, &
             ma86_finalise
   public :: ma86_get_n__
   LOGICAL, PUBLIC, PARAMETER :: ma86_available = .FALSE.
   private :: ip_, long_, lp_, rp_

   ! Numerical constants
   real(rp_), parameter, private :: one  = 1.0_rp_
   real(rp_), parameter, private :: zero = 0.0_rp_

   ! Default values
   integer(ip_),  parameter, private :: nemin_default = 32
     ! node amalgamation parameter
   integer(ip_),  parameter, private :: nb_default = 256
     ! Block size with dense kernel
   integer(ip_),  parameter :: nbi_default = 16 ! Default inner block size.
   integer(ip_),  parameter, private :: pool_default = 25000
     ! size of task pool

   ! Symbolic constants
   ! These flag the different tasks within factor and solve
   integer(ip_),  parameter, private :: TASK_DONE             = -1
   integer(ip_),  parameter, private :: TASK_NONE             = 0
   integer(ip_),  parameter, private :: TASK_FACTORIZE_COLUMN = 2
   integer(ip_),  parameter, private :: TASK_UPDATE_INTERNAL  = 3
   integer(ip_),  parameter, private :: TASK_UPDATE_BETWEEN   = 4
   integer(ip_),  parameter, private :: TASK_SLV_FSLV         = 6
     ! Fwds solve on diag block
   integer(ip_),  parameter, private :: TASK_SLV_FUPD         = 7
     ! Fwds update in solve
   integer(ip_),  parameter, private :: TASK_SLV_BSLV         = 8
     ! Bwds solve on diag block
   integer(ip_),  parameter, private :: TASK_SLV_BUPD         = 9
     ! Bwds update in solve

   ! Types of solve job
   integer(ip_),  parameter :: SOLVE_JOB_ALL         = 0
   integer(ip_),  parameter :: SOLVE_JOB_FWD         = 1
   integer(ip_),  parameter :: SOLVE_JOB_D           = 2
   integer(ip_),  parameter :: SOLVE_JOB_BWD         = 3
   integer(ip_),  parameter :: SOLVE_JOB_D_AND_BWD   = 4

   ! How processors share cache                    Example
   integer(ip_),  parameter, private :: CACHE_COMPACT       = 1
     ! [0,1], [2,3], [4,5], [6,7]
   integer(ip_),  parameter, private :: CACHE_SCATTER       = 2
     ! [0,4]. [1,5], [2,6], [3,7]
   integer(ip_),  parameter, private :: CACHE_IDENTITY      = 3
     ! 0, 1, 2, 3, 4, 5, 6, 7

   ! Error flags
   integer(ip_),  parameter, private :: MA86_SUCCESS               = 0
   integer(ip_),  parameter, private :: MA86_ERROR_ALLOCATION      = -1
   integer(ip_),  parameter, private :: MA86_ERROR_ORDER           = -2
   integer(ip_),  parameter, private :: MA86_ERROR_SINGULAR        = -3
   integer(ip_),  parameter, private :: MA86_ERROR_X_SIZE          = -4
   integer(ip_),  parameter, private :: MA86_ERROR_INFINITY        = -5
   integer(ip_),  parameter, private :: MA86_ERROR_JOB_OOR         = -6
   integer(ip_),  parameter, private :: MA86_ERROR_STATIC_SMALL    = -7
   integer(ip_),  parameter, private :: MA86_ERROR_UNKNOWN         = -99

   ! warning flags
   integer(ip_),  parameter, private :: MA86_WARNING_POOL_SMALL       = 1
   integer(ip_),  parameter, private :: MA86_WARNING_SINGULAR         = 2
   integer(ip_),  parameter, private :: MA86_WARNING_POOL_SIN         = 3

   interface MA86_analyse
      module procedure MA86_analyse_real
   end interface

   interface MA86_factor
      module procedure MA86_factor_real
   end interface

   interface MA86_factor_solve
      module procedure MA86_factor_solve_one_real, &
                       MA86_factor_solve_mult_real
   end interface

   interface MA86_solve
      module procedure MA86_solve_one_real, MA86_solve_mult_real
   end interface

   interface MA86_finalise
      module procedure MA86_finalise_real
   end interface

   interface MA86_get_n__
      module procedure ma86_get_n_real
   end interface MA86_get_n__

   type block_type
      integer(ip_) :: bcol ! block column that blk belongs to
      integer(ip_) :: blkm ! height of block (number of rows in blk)
      integer(ip_) :: blkn ! width of block (number of columns in blk)
      integer(long_) :: dblk ! id of the block on the diagonal within the
         ! block column to which blk belongs
      integer(ip_) :: dep_initial ! initial dependency count for block,
         ! In indef case, countdown occurs in bcol on a block column basis
      integer(long_) :: id        ! The block identitifier (ie, its number blk)
      integer(long_) :: last_blk  ! id of the last block within the
         ! block column to which blk belongs
      integer(ip_) :: node            ! node to which blk belongs
      integer(ip_) :: sa              ! posn of the first entry of the
         ! block blk within the array that holds the block column of L
         ! that blk belongs to.
      logical(lp_) :: touched ! is this the first time block is touched
      integer(ip_) :: sa_new          ! posn of the first entry of the
         ! block blk within the array that holds the block column of L
         ! that blk belongs to, after delays have been allowed for.
         ! This is computed during factorize.
!$    integer(omp_lock_kind) :: lock   ! Lock for altering dep
!$    integer(omp_lock_kind) :: alock  ! Lock for
         ! for this block.
         ! Note: locks initialised in ma86_analyse and destroyed
         !       in ma86_finalise
   end type block_type

   type node_type
      integer(long_) :: blk_sa ! identifier of the first block in node
      integer(long_) :: blk_en ! identifier of the last block in node
      integer(ip_) :: nb ! Block size for nodal matrix
        ! If number of cols nc in nodal matrix is less than control%nb but
        ! number of rows is large, the block size for the node is taken as
        ! control%nb**2/nc, rounded up to a multiple of 8. The aim is for
        ! the numbers of entries in the blocks to be similar to those in the
        ! normal case.
      integer(ip_) :: sa ! index (in pivotal order) first column of the node
      integer(ip_) :: en ! index (in pivotal order) last column of the node
      integer(ip_),  allocatable :: index(:) ! holds the permuted variable
        ! list for node. They are sorted into increasing order.
        ! index is set up by ma86_analyse
      integer(ip_) :: nchild ! number of children node has in assembly tree
      integer(ip_),  allocatable :: child(:) ! holds children of node
      integer(ip_) :: num_delay = 0 ! number of delayed eliminations to
         ! pass to parent
      integer(ip_) :: parent ! Parent of node in assembly tree
      integer(ip_) :: least_desc ! Least descendant in assembly tree
      ! next pointer in linked list of delay sources
      integer(ip_) :: delay_next
   end type node_type

   type thread_info
      integer(ip_) :: num_delay    = 0 ! number of delayed pivots
      integer(long_) :: num_flops  = 0 ! number of floating point operations
      integer(long_) :: num_factor = 0 ! number of entries in factors
      integer(ip_) :: num_neg      = 0 ! number of negative pivots in the real
                                   ! or complex Hermitian case.
      integer(ip_) :: num_nothresh  = 0 ! number of pivots not satisfying
        ! pivot threshold test with control%u
      integer(ip_) :: num_perturbed = 0 ! number of perturbed pivots
      integer(ip_) :: num_two = 0 ! number of 2x2 pivots
      integer(ip_) :: num_zero_pivots  = 0 ! number of zero pivots
      real (rp_) :: usmall = -one ! Set to zero if num_perturbed > 0.
         ! Otherwise, if q < p, it holds the value of cntl%umin that
         ! would have led to a greater value of q and if q = p, it holds
         ! the smallest relative pivot value of the chosen pivots.
      real(rp_) :: detlog = zero ! logarithm of abs value of det A
      integer(ip_) :: detsign = 1 ! in the real or complex Hermitian case,
                                  ! holds sign of determinant or 0 if A is
                                  ! singular
   end type thread_info

   type lfactor
      logical(lp_) :: use_new ! set to .true. for the current
        ! block col if lcol_new being used rather than lcol
      integer(ip_) :: blkn_new ! number of columns in lcol_new
      integer(long_) :: dblk      ! identifier of diagonal block in block col.
      integer(ip_) :: dep             ! dependency count
      integer(ip_) :: local ! local index of block column within node.
      integer(ip_) :: nelim ! number of eliminations performed in block column.
      integer(ip_) :: nrow ! # rows in block column at start of factorize
         ! ie number of rows in lcol. Number of rows in lcol_new is
         ! nrow + num_delay   (num_delay = blkn_new - blkn, where blkn is no.
         ! cols in lcol and can be found from blocks using dblk)
      integer(ip_) :: col ! start position of bcol in rhs vector for solve
      integer(ip_) :: delay_head
      integer(ip_),  allocatable :: index_new(:) ! holds index list within
         ! factorize (will include delays)
!$    integer(omp_lock_kind) :: lock   ! lock so only one thread at a time
         ! can alter the block column
      real(rp_), dimension(:), allocatable :: lcol ! holds block column
      real(rp_), dimension(:), allocatable :: lcol_new ! holds block column
         ! after delayed cols have been added (only used if
         ! there are delays to accommodate in current block col)
      real(rp_) , dimension(:), allocatable :: d ! holds block of D.
   end type lfactor

   type slv_count_type
      integer(ip_) :: dep
      integer(ip_) :: dblk
!$    integer(kind=omp_lock_kind) :: lock
   end type slv_count_type

   type MA86_control
      logical(lp_) :: action          = .true. ! Do we keep
         ! going even if matrix is singular (abort if .false.)
      integer(ip_) :: diagnostics_level = 0      ! Controls diagnostic printing.
         ! Possible values are:
         !  < 0: no printing.
         !    0: error and warning messages only.
         !    1: as 0 plus basic diagnostic printing.
         !    2: as 1 plus some more detailed diagnostic messages.
         !    3: as 2 plus all entries of user-supplied arrays.
      integer(ip_) :: nb    = nb_default ! Controls the size of the
         ! blocks used within each node (used to set nb within node_type)
      integer(ip_) :: nbi   = nbi_default ! Inner block size for use with ma64
      integer(ip_) :: nemin = nemin_default
         ! Node amalgamation parameter. A child node is merged with its parent
         ! if they both involve fewer than nemin eliminations.
      integer(ip_) :: pool_size       = pool_default ! Size of task pool arrays
      real(rp_) :: small          = 1e-20 ! Pivots less than small are
         ! treated as zero
      real(rp_) :: static         = zero ! Control static pivoting
      real(rp_) :: u              = 0.01 ! Pivot tolerance
      real(rp_) :: umin           = one  ! Minimum pivot tolerance
      integer(ip_) :: unit_diagnostics = 6    ! unit for diagnostic messages
         ! Printing is suppressed if unit_diagnostics  <  0.
      integer(ip_) :: unit_error       = 6    ! unit for error messages
         ! Printing is suppressed if unit_error  <  0.
      integer(ip_) :: unit_warning     = 6    ! unit for warning messages
         ! Printing is suppressed if unit_warning  <  0.
      integer(ip_) :: scaling          = 1    ! scaling routine to use
         ! 0 = none or user defined (user iff scale is present)
         ! 1 = mc64
      integer(ip_) :: cache_tq_sz     = 100    ! Size of local task stack
      integer(ip_) :: cache_layout    = CACHE_COMPACT ! Proc <=> cache mapping
      integer(ip_) :: cache_cores     = 2      ! Number of cores per cache
      integer(ip_) :: min_width_blas  = 8      ! Minimum width of source block
         ! before we use an indirect update_between

   end type MA86_control

   type MA86_info
      real(rp_) :: detlog = zero         ! Holds logarithm of abs det A (or 0)
      integer(ip_) :: detsign = 0            ! Holds sign of determinant +/-1,0
      integer(ip_) :: flag = 0               ! Error return flag (0 on success)
      integer(ip_) :: matrix_rank = 0        ! Rank of matrix
      integer(ip_) :: maxdepth = 0           ! Maximum depth of the tree.
      integer(ip_) :: num_delay = 0          ! Number of delayed pivots
      integer(long_) :: num_factor = 0_long_ ! Number of entries in the factor.
      integer(long_) :: num_flops = 0_long_  ! Number of flops for factor.
      integer(ip_) :: num_nodes = 0          ! Number of nodes
!     integer(ip_) :: num_sup = 0            ! Number of supervariables
      integer(ip_) :: num_two = 0            ! Number of 2x2 pivots
      integer(ip_) :: num_neg = 0            ! Number of negative pivots
      integer(ip_) :: num_nothresh = 0       ! Number of pivots not satisfying u
      integer(ip_) :: num_perturbed = 0      ! Number of perturbed pivots
      integer(ip_) :: pool_size = pool_default  ! Maximum size of task pool used
      integer(ip_) :: stat = 0               ! STAT value on error return -1.
      real(rp_) :: usmall = zero         ! smallest threshold parameter used
   end type MA86_info

   type ma86_keep
      type(zd11_type) :: a ! Holds lower and upper triangular parts of A
      type(block_type), dimension(:), allocatable :: blocks ! block info
      integer(ip_),  dimension(:), allocatable :: flag_array ! allocated to
        ! have size equal to the number of threads. For each thread, holds
        ! error flag
      integer(long_) :: final_blk = 0 ! Number of blocks. Used for destroying
        ! locks in finalise
      type(ma86_info) :: info ! Holds copy of info
      integer(ip_) :: maxmn ! holds largest block dimension
      integer(ip_) :: n  ! Order of the system.
      type(node_type), dimension(:), allocatable :: nodes ! nodal info
      integer(ip_) :: nbcol = 0 ! number of block columns in L
      type(lfactor), dimension(:), allocatable :: lfact
         ! holds block cols of L
   end type ma86_keep

   type dagtask
      integer(ip_) :: task_type    ! One of TASK_FACTORIZE_COLUMN, ...
      integer(long_) :: dest   ! id of the target (destination) block
      integer(long_) :: src1   !
         ! if task_type = TASK_UPDATE_INTERNAL, src1 holds the id of the first
         ! source block
         ! if task_type = TASK_UPDATE_BETWEEN, src1 holds the id of a block
         ! in the block column of the source node that is used
         ! in updating dest.
      integer(long_) :: src2
         ! if task_type = TASK_UPDATE_INTERNAL, src2 holds the id of the second
         ! source block
         ! (src1 and src2 are blocks belonging to the same block column
         ! of the source node with src1 .le. src2)
         ! src2 is not used by the other tasks
      integer(ip_) :: csrc(2)
      integer(ip_) :: rsrc(2)
         ! for an UPDATE_BETWEEN task, we need to hold some additional
         ! information, which locates the source blocks rsrc and csrc
         ! within the source block col.
         ! This info. is set up the subroutine add_between_updates
   end type dagtask

   type taskstack
      integer(ip_) :: max_pool_size = 0 ! max. number of tasks that are in
         ! the task pool at any one time during the factorization.
      logical(lp_) :: abort = .false.   ! true if we have aborted
      integer(ip_) :: active ! Number of active threads
         ! (number of tasks in execution)
      type(dagtask), dimension(:,:), allocatable :: ctasks ! local task stacks.
         ! allocated to have size (control%cache_tq_sz, ncache), where
         ! ncache is number of caches
      integer(ip_),  dimension(:), allocatable :: cheads   ! Heads local stacks
         ! allocated to have size equal to number of caches
!$    integer(omp_lock_kind), dimension(:), allocatable :: clocks
         ! Locks for local stacks.
      integer(ip_) :: freehead  ! Holds the head of linked list of
         ! entries in the task pool that are free
!$    integer(omp_lock_kind) :: lock   ! lock so only one thread at a time
         ! can read/alter the task pool
      integer(ip_) :: lowest_priority_value = huge(0) !
         ! lowest priority value of the tasks in the pool.
         ! The priority value for each of the different types of task is
         !  1. factor             Highest priority
         !  2. solve
         !  3. update_internal
         !  4. update_between     Lowest priority
      integer(ip_),  dimension(:), allocatable :: next  ! next task in linked
         !  list. allocated to have size pool_size. Reallocated if initial
         !  setting for pool_size found to be too small.
      integer(ip_) :: pool_size   ! sizes of task pool arrays next and tasks.
         ! Initialised to control%pool_size
      integer(ip_) :: prihead(4)  ! Holds the heads of the linked lists for
         ! tasks with priority values 1,2,3,4.
      type(dagtask), dimension(:), allocatable :: tasks ! Holds tasks.
         ! allocated to have size pool_size. Reallocated if initial setting
         ! for pool_size found to be too small.
      integer(ip_) :: total       ! Total number of tasks in pool
   end type taskstack

contains

subroutine MA86_analyse_real(n, ptr, row, order, keep, control, info)
   integer(ip_),  intent(in) :: n ! order of A
   integer(ip_),  intent(in) :: row(:) ! row indices of lower triangular part
   integer(ip_),  intent(in) :: ptr(:) ! col pointers for lower triangular part
   integer(ip_),  intent(inout), dimension(:) :: order
      ! order(i) must hold position of i in the pivot sequence.
      ! On exit, holds the pivot order to be used by MA86_factor.
   ! For details of keep, control, info : see derived type descriptions
   type(MA86_keep) :: keep
   type(MA86_control), intent(in) :: control
   type(MA86_info), intent(inout) :: info

        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MA86_analyse with its HSL namesake ', /,                  &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

   info%flag = GALAHAD_unavailable_option
   info%num_factor = 0_long_
   info%num_flops = 0_long_
   info%num_nodes = 0
   info%maxdepth = 0
   info%stat = 0
end subroutine MA86_analyse_real

subroutine MA86_factor_real(n, ptr, row, val, order, keep, control, info)
   integer(ip_),  intent(in) :: n ! order of A
   integer(ip_),  intent(in) :: row(:) ! row indices of lower triangular part
   integer(ip_),  intent(in) :: ptr(:) ! col pointers for lower triangular part
   real(rp_), intent(in) :: val(:) ! matrix values
   integer(ip_),  intent(in) :: order(:) ! holds pivot order (must be unchanged
      ! since the analyse phase)
   type(MA86_keep), intent(inout) :: keep ! see description of derived type
   type(MA86_control), intent(in) :: control ! see description of derived type
   type(MA86_info) :: info ! see description of derived type

        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MA86_factor with its HSL namesake ', /,                   &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

   info%flag = GALAHAD_unavailable_option
   info%num_factor = 0_long_
   info%num_flops = 0_long_
   info%num_nodes = 0
   info%maxdepth = 0
   info%stat = 0

end subroutine MA86_factor_real

subroutine MA86_factor_solve_one_real(n, ptr, row, val, order, keep, control,&
      info, x)
   integer(ip_),  intent(in) :: n ! order of A
   integer(ip_),  intent(in) :: row(:) ! row indices of lower triangular part
   integer(ip_),  intent(in) :: ptr(:) ! col pointers for lower triangular part
   real(rp_), intent(in) :: val(:) ! matrix values
   integer(ip_),  intent(in) :: order(:) ! holds pivot order (must be unchanged
      ! since the analyse phase)
   type(MA86_keep), intent(inout) :: keep ! see description of derived type
   type(MA86_control), intent(in) :: control ! see description of derived type
   type(MA86_info) :: info ! see description of derived type
   real(rp_), intent(inout) :: x(keep%n) ! On entry, x must
      ! be set so that if i has been used to index a variable,
      ! x(i) is the corresponding component of the right-hand side.
      ! On exit, if i has been used to index a variable,
      ! x(i) holds solution for variable i.

        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MA86_solve_one with its HSL namesake ', /,               &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

   info%flag = GALAHAD_unavailable_option
   info%num_factor = 0_long_
   info%num_flops = 0_long_
   info%num_nodes = 0
   info%maxdepth = 0
   info%stat = 0
end subroutine MA86_factor_solve_one_real

subroutine MA86_factor_solve_mult_real(n, ptr, row, val, order, keep, &
      control, info, nrhs, lx, x)
   integer(ip_),  intent(in) :: n ! order of A
   integer(ip_),  intent(in) :: row(:) ! row indices of lower triangular part
   integer(ip_),  intent(in) :: ptr(:) ! col pointers for lower triangular part
   real(rp_), intent(in) :: val(:) ! matrix values
   integer(ip_),  intent(in) :: order(:) ! holds pivot order (must be unchanged
      ! since the analyse phase)
   type(MA86_keep), intent(inout) :: keep ! see description of derived type
   type(MA86_control), intent(in) :: control ! see description of derived type
   type(MA86_info) :: info ! see description of derived type
   integer(ip_),  intent(in) :: nrhs ! number of right-hand sides to solver for
   integer(ip_),  intent(in) :: lx ! first dimension of x
   real(rp_), intent(inout) :: x(lx,nrhs) ! On entry, x must
      ! be set so that if i has been used to index a variable,
      ! x(i,j) is the corresponding component of the
      ! right-hand side for the jth system (j = 1,2,..., nrhs).
      ! On exit, if i has been used to index a variable,
      ! x(i,j) holds solution for variable i to system j

        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MA86_solve_mult with its HSL namesake ', /,               &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

   info%flag = GALAHAD_unavailable_option
   info%num_factor = 0_long_
   info%num_flops = 0_long_
   info%num_nodes = 0
   info%maxdepth = 0
   info%stat = 0
end subroutine MA86_factor_solve_mult_real

subroutine MA86_solve_one_real(x,order,keep,control,info,job)
   type(MA86_keep), intent(inout) :: keep
   real(rp_), intent(inout) :: x(keep%n) ! On entry, x must
      ! be set so that if i has been used to index a variable,
      ! x(i) is the corresponding component of the right-hand side.
      ! On exit, if i has been used to index a variable,
      ! x(i) holds solution for variable i.
   integer(ip_),  intent(in) :: order(:) ! pivot order. must be unchanged
   ! For details of keep, control, info : see derived type description
   type(MA86_control), intent(in) :: control
   type(MA86_info) :: info
   integer(ip_),  optional, intent(in) :: job  ! used to indicate whether
      ! partial solution required
      ! job = 0 or absent: complete solve performed
      ! job = 1 : forward eliminations only (PLx = b)
      ! job = 2 : backsubs only ((PL)^Tx = b)

        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MA86_solve_one with its HSL namesake ', /,                &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

   info%flag = GALAHAD_unavailable_option
   info%num_factor = 0_long_
   info%num_flops = 0_long_
   info%num_nodes = 0
   info%maxdepth = 0
   info%stat = 0
end subroutine MA86_solve_one_real

subroutine MA86_solve_mult_real(nrhs,lx,x,order,keep, control,info,job)
   integer(ip_),  intent(in) :: nrhs ! number of right-hand sides to solver for
   integer(ip_),  intent(in) :: lx ! first dimension of x
   real(rp_), intent(inout) :: x(lx,nrhs) ! On entry, x must
      ! be set so that if i has been used to index a variable,
      ! x(i,j) is the corresponding component of the
      ! right-hand side for the jth system (j = 1,2,..., nrhs).
      ! On exit, if i has been used to index a variable,
      ! x(i,j) holds solution for variable i to system j
   integer(ip_),  intent(in) :: order(:) ! pivot order. must be unchanged
   ! For details of keep, control, info : see derived type description
   type(MA86_keep), intent(inout) :: keep
   type(MA86_control), intent(in) :: control
   type(MA86_info) :: info
   integer(ip_),  optional, intent(in) :: job  ! used to indicate whether
      ! partial solution required
      ! job = 0 or absent: complete solve performed
      ! job = 1 : forward eliminations only (PLX = B).
      ! job = 2 : backsubs only ((PL)^TX = B)

        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MA86_solve_mult with its HSL namesake ', /,               &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

   info%flag = GALAHAD_unavailable_option
   info%num_factor = 0_long_
   info%num_flops = 0_long_
   info%num_nodes = 0
   info%maxdepth = 0
   info%stat = 0
end subroutine MA86_solve_mult_real

subroutine MA86_finalise_real(keep, control)
   type(MA86_keep), intent(inout) :: keep    ! See derived-type declaration
   type(MA86_control), intent(in) :: control ! See derived-type declaration

        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MA86_finalize with its HSL namesake ', /,                 &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

end subroutine MA86_finalise_real

subroutine factorize_posdef(a, order, keep, control, info, nrhs, ldr, rhs)
   type(zd11_type), intent(in) :: a ! System matrix in CSC format.
   integer(ip_),  intent(in) :: order(:)  ! holds pivot order
   type(MA86_keep), intent(inout) :: keep ! see description of derived type
   type(MA86_control), intent(in) :: control ! see description of derived type
   type(MA86_info), intent(inout) :: info ! see description of derived type
   integer(ip_),  intent(in) :: nrhs  ! number of right-hand sides (maybe = 0)
   integer(ip_),  intent(in) :: ldr  ! leading extent of rhs
   real(rp_), intent(inout) :: rhs(ldr*nrhs)  ! On entry holds rhs data.

end subroutine factorize_posdef

subroutine ma86_print_flag(iflag, control, context, st)
   integer(ip_),  intent(in) :: iflag
   type(ma86_control), intent(in) :: control
   integer(ip_),  intent(in), optional :: st
   ! context: is an optional assumed size character array of intent(in).
   ! It describes the context under which the error occured
   character (len=*), optional, intent(in) :: context

        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MA86_print_flag with its HSL namesake ', /,               &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

end subroutine MA86_print_flag

pure integer(ip_) function ma86_get_n_real(keep)
   type(ma86_keep), intent(in) :: keep
   ma86_get_n_real = keep%n
end function ma86_get_n_real

end module hsl_ma86_real
