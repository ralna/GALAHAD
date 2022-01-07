! THIS VERSION: GALAHAD 4.0 - 2022-01-07 AT 13:00 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M A 8 7    M O D U L E  -*-*-*-
module hsl_MA87_double
!$ use omp_lib
   use hsl_mc78_integer
   use hsl_mc34_double
   use hsl_zd11_double

   implicit none
   public :: ma87_get_n__

   ! Parameters (all private)
   ! Data kinds
   integer, parameter, private :: wp   = kind(0d0)
   integer, parameter, private :: long = selected_int_kind(18)
   integer, parameter, private :: short = kind(0)

   ! Numerical constants
   real(wp), parameter, private :: one  = 1.0_wp
   real(wp), parameter, private :: zero = 0.0_wp

   ! Default values
   integer, parameter, private :: nemin_default = 32
     ! node amalgamation parameter
   integer, parameter, private :: nb_default = 256
     ! Block size with dense kernel
   integer, parameter, private :: pool_default = 25000
     ! size of task pool

   ! Symbolic constants
   ! These flag the different tasks within factor and solve
   integer, parameter, private :: TASK_DONE             = -1
   integer, parameter, private :: TASK_NONE             = 0
   integer, parameter, private :: TASK_FACTORIZE_BLOCK  = 1
   integer, parameter, private :: TASK_UPDATE_INTERNAL  = 3
   integer, parameter, private :: TASK_UPDATE_BETWEEN   = 4
   integer, parameter, private :: TASK_SOLVE_BLOCK      = 5
   integer, parameter, private :: TASK_SLV_FSLV         = 6
     ! Fwds solve on diag block
   integer, parameter, private :: TASK_SLV_FUPD         = 7
     ! Fwds update in solve
   integer, parameter, private :: TASK_SLV_BSLV         = 8
     ! Bwds solve on diag block
   integer, parameter, private :: TASK_SLV_BUPD         = 9
     ! Bwds update in solve

   ! Types of solve job
   integer, parameter, private :: SOLVE_JOB_ALL         = 0
   integer, parameter, private :: SOLVE_JOB_FWD         = 1
   integer, parameter, private :: SOLVE_JOB_BWD         = 2
   ! How processors share cache                    Example
   integer, parameter, private :: CACHE_COMPACT       = 1
     ! [0,1], [2,3], [4,5], [6,7]
   integer, parameter, private :: CACHE_SCATTER       = 2
     ! [0,4]. [1,5], [2,6], [3,7]
   integer, parameter, private :: CACHE_IDENTITY      = 3
     ! 0, 1, 2, 3, 4, 5, 6, 7

   ! Error flags
   integer, parameter, private :: MA87_SUCCESS               = 0
   integer, parameter, private :: MA87_ERROR_ALLOCATION      = -1
   integer, parameter, private :: MA87_ERROR_ORDER           = -2
   integer, parameter, private :: MA87_ERROR_NOT_POSDEF      = -3
   integer, parameter, private :: MA87_ERROR_X_SIZE          = -4
   integer, parameter, private :: MA87_ERROR_INFINITY        = -5
   integer, parameter, private :: MA87_ERROR_JOB_OOR         = -6
   integer, parameter, private :: MA87_ERROR_UNKNOWN         = -99

   ! warning flags
   integer, parameter, private :: MA87_WARNING_POOL_SMALL    = 1

   interface MA87_analyse
      module procedure MA87_analyse_double
   end interface

   interface MA87_factor
      module procedure MA87_factor_double
   end interface

   interface MA87_factor_solve
      module procedure MA87_factor_solve_one_double, &
                       MA87_factor_solve_mult_double
   end interface

   interface MA87_solve
      module procedure MA87_solve_one_double, MA87_solve_mult_double
   end interface

   interface MA87_sparse_fwd_solve
      module procedure MA87_sparse_fwd_solve_double
   end interface

   interface MA87_finalise
      module procedure MA87_finalise_double
   end interface

   interface ma87_get_n__
      module procedure ma87_get_n_double
   end interface ma87_get_n__

   type block_type
      ! Static info, which is set in ma87_analayse
      integer :: bcol            ! block column that blk belongs to
      integer :: blkm            ! height of block (number of rows in blk)
      integer :: blkn            ! width of block (number of columns in blk)
      integer(long) :: dblk      ! id of the block on the diagonal within the
         ! block column to which blk belongs
      integer :: dep_initial     ! initial dependency count for block,
      integer(long) :: id        ! The block identitifier (ie, its number blk)
      integer(long) :: last_blk  ! id of the last block within the
         ! block column to which blk belongs
      integer :: node            ! node to which blk belongs
      integer :: sa              ! posn of the first entry of the
         ! block blk within the array that holds the block column of L
         ! that blk belongs to.

      ! Non-static info
      integer :: dep  ! dependency countdown/marker. Once factor or solve done,
                      ! value is -2.
!$    integer(omp_lock_kind) :: lock   ! Lock for altering dep
!$    integer(omp_lock_kind) :: alock  ! Lock for altering values in keep%lfact
         ! for this block.
         ! Note: locks initialised in ma87_analyse and destroyed
         !       in ma87_finalise
   end type block_type

   type node_type
      integer(long) :: blk_sa ! identifier of the first block in node
      integer(long) :: blk_en ! identifier of the last block in node

      integer :: nb ! Block size for nodal matrix
        ! If number of cols nc in nodal matrix is less than control%nb but
        ! number of rows is large, the block size for the node is taken as
        ! control%nb**2/nc, rounded up to a multiple of 8. The aim is for
        ! the numbers of entries in the blocks to be similar to those in the
        ! normal case.

      integer :: sa ! index (in pivotal order) of the first column of the node
      integer :: en ! index (in pivotal order) of the last column of the node

      integer, allocatable :: index(:) ! holds the permuted variable
        ! list for node. They are sorted into increasing order.
        ! index is set up by ma87_analyse

      integer :: nchild ! number of children node has in assembly tree
      integer, allocatable :: child(:) ! holds children of node
      integer :: num_delay = 0 ! number of delayed eliminations to
         ! pass to parent
      integer :: parent ! Parent of node in assembly tree
      integer :: least_desc ! Least descendant in assembly tree
   end type node_type

   type lfactor
      real(wp), dimension(:), allocatable :: lcol ! holds block column
   end type lfactor

   type lmap_type
      integer(long) :: len_map ! length of map
      integer(long), allocatable :: map(:,:) ! holds map from user's val
         ! array into lfact(:)%lcol values
   end type lmap_type

   type slv_count_type
      integer :: dep
      integer :: dblk
!$    integer(kind=omp_lock_kind) :: lock
   end type slv_count_type

   type MA87_control
      integer :: diagnostics_level = 0      ! Controls diagnostic printing.
         ! Possible values are:
         !  < 0: no printing.
         !    0: error and warning messages only.
         !    1: as 0 plus basic diagnostic printing.
         !    2: as 1 plus some more detailed diagnostic messages.
         !    3: as 2 plus all entries of user-supplied arrays.
      integer :: nb    = nb_default ! Controls the size of the
         ! blocks used within each node (used to set nb within node_type)
      integer :: nemin = nemin_default
         ! Node amalgamation parameter. A child node is merged with its parent
         ! if they both involve fewer than nemin eliminations.
      integer :: pool_size       = pool_default ! Size of task pool arrays
      integer :: unit_diagnostics = 6    ! unit for diagnostic messages
         ! Printing is suppressed if unit_diagnostics  <  0.
      integer :: unit_error       = 6    ! unit for error messages
         ! Printing is suppressed if unit_error  <  0.
      integer :: unit_warning     = 6    ! unit for warning messages
         ! Printing is suppressed if unit_warning  <  0.
      integer :: cache_tq_sz     = 10!100    ! Size of local task stack
      integer :: cache_layout    = CACHE_COMPACT ! Proc <=> cache mapping
      integer :: cache_cores     = 2      ! Number of cores per cache
      integer :: min_width_blas  = 8      ! Minimum width of source block
         ! before we use an indirect update_between
      real(wp) :: diag_zero_plus = 0.0
      real(wp) :: diag_zero_minus = 0.0
     ! There are three cases for diagonal entries d_ii:
      ! diag_zero_plus  < d_ii                                  +ive eigenvalue
      ! diag_zero_minus < d_ii <= diag_zero_plus                zero eigenvalue
      !            d_ii < min(diag_zero_minus, diag_zero_plus)  -ive eigenvalue
      ! Traditional LAPACK potrf() corresponds to
      ! diag_zero_plus = diag_zero_minus = 0.0
   end type MA87_control

   type MA87_info
      real(wp) :: detlog = 0            ! Holds logarithm of abs det A (or 0)
      integer :: flag = 0               ! Error return flag (0 on success)
      integer :: maxdepth = 0           ! Maximum depth of the tree.
      integer(long) :: num_factor = 0_long ! Number of entries in the factor.
      integer(long) :: num_flops = 0_long  ! Number of flops for factor.
!     integer :: num_sup = 0            ! Number of supervariables
      integer :: num_nodes = 0          ! Number of nodes
      integer :: pool_size = pool_default  ! Maximum size of task pool used
      integer :: stat = 0               ! STAT value on error return -1.
      integer :: num_zero = 0           ! Num pivots in range [diag_zero, 0.0]
   end type MA87_info

   type ma87_keep
      type(zd11_type) :: a ! Holds lower and upper triangular parts of A
      type(block_type), dimension(:), allocatable :: blocks ! block info
      integer, dimension(:), allocatable :: flag_array ! allocated to
        ! have size equal to the number of threads. For each thread, holds
        ! error flag
      integer(long) :: final_blk = 0 ! Number of blocks. Used for destroying
        ! locks in finalise
      type(ma87_info) :: info ! Holds copy of info
      integer :: maxmn ! holds largest block dimension
      integer :: n  ! Order of the system.
      type(node_type), dimension(:), allocatable :: nodes ! nodal info
      integer :: nbcol = 0 ! number of block columns in L
      type(lfactor), dimension(:), allocatable :: lfact
         ! holds block cols of L
      type(lmap_type), dimension(:), allocatable :: lmap
         ! holds mapping from matrix values into lfact
      logical, dimension(:), allocatable :: zero_flag
         ! true if variable i was in range [diag_zero, 0.0] upon pivoting
   end type ma87_keep

   type dagtask
      integer :: task_type    ! One of TASK_FACTORIZE_BLOCK, ...
      integer(long) :: dest   ! id of the target (destination) block
      integer(long) :: src1   !
         ! if task_type = TASK_UPDATE_INTERNAL, src1 holds the id of the first
         ! source block
         ! if task_type = TASK_UPDATE_BETWEEN, src1 holds the id of a block
         ! in the block column of the source node that is used
         ! in updating dest.
      integer(long) :: src2
         ! if task_type = TASK_UPDATE_INTERNAL, src2 holds the id of the second
         ! source block
         ! (src1 and src2 are blocks belonging to the same block column
         ! of the source node with src1 .le. src2)
         ! src2 is not used by the other tasks
      integer :: csrc(2)
      integer :: rsrc(2)
         ! for an UPDATE_BETWEEN task, we need to hold some additional
         ! information, which locates the source blocks rsrc and csrc
         ! within the source block col.
         ! This info. is set up the subroutine add_between_updates
   end type dagtask

   type taskstack
      integer :: max_pool_size = 0 ! max. number of tasks that are in
         ! the task pool at any one time during the factorization.
      logical :: abort = .false.   ! true if we have aborted
      integer :: active ! Number of active threads
         ! (number of tasks in execution)
      type(dagtask), dimension(:,:), allocatable :: ctasks ! local task stacks.
         ! allocated to have size (control%cache_tq_sz, ncache), where
         ! ncache is number of caches
      integer, dimension(:), allocatable :: cheads   ! Heads for local stacks.
         ! allocated to have size equal to number of caches
!$    integer(omp_lock_kind), dimension(:), allocatable :: clocks
         ! Locks for local stacks.
      integer :: freehead  ! Holds the head of linked list of
         ! entries in the task pool that are free
!$    integer(omp_lock_kind) :: lock   ! lock so only one thread at a time
         ! can read/alter the task pool
      integer :: lowest_priority_value = huge(0) !
         ! lowest priority value of the tasks in the pool.
         ! The priority value for each of the different types of task is
         !  1. factor             Highest priority
         !  2. solve
         !  3. update_internal
         !  4. update_between     Lowest priority
      integer, dimension(:), allocatable :: next  ! next task in linked list.
         ! allocated to have size pool_size. Reallocated if initial setting
         ! for pool_size found to be too small.
      integer :: pool_size   ! sizes of task pool arrays next and tasks.
         ! Initialised to control%pool_size
      integer :: prihead(4)  ! Holds the heads of the linked lists for tasks
         ! with priority values 1,2,3,4.
      type(dagtask), dimension(:), allocatable :: tasks ! Holds tasks.
         ! allocated to have size pool_size. Reallocated if initial setting
         ! for pool_size found to be too small.
      integer :: total       ! Total number of tasks in pool
   !**   real, dimension(:), allocatable :: waiting  ! Allocated to have size
   !**  ! equal to the number of threads. Used to hold times the threads spent
   !**  ! waiting if control%time_out >= 0

   end type taskstack

contains

subroutine MA87_analyse_double(n, ptr, row, order, keep, control, info)
   USE GALAHAD_SYMBOLS
   integer, intent(in) :: n ! order of A
   integer, intent(in) :: row(:) ! row indices of lower triangular part
   integer, intent(in) :: ptr(:) ! col pointers for lower triangular part
   integer, intent(inout), dimension(:) :: order
      ! order(i) must hold position of i in the pivot sequence.
      ! On exit, holds the pivot order to be used by MA87_factor.
   ! For details of keep, control, info : see derived type descriptions
   type(MA87_keep) :: keep
   type(MA87_control), intent(in) :: control
   type(MA87_info), intent(inout) :: info

        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MA87_analyse with its HSL namesake ', /,                  &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

   info%flag = GALAHAD_unavailable_option
   info%num_factor = 0_long
   info%num_flops = 0_long
   info%num_nodes = 0
   info%maxdepth = 0
   info%stat = 0
end subroutine MA87_analyse_double

subroutine MA87_factor_double(n, ptr, row, val, order, keep, control, info)
   USE GALAHAD_SYMBOLS
   integer, intent(in) :: n ! order of A
   integer, intent(in) :: row(:) ! row indices of lower triangular part
   integer, intent(in) :: ptr(:) ! col pointers for lower triangular part
   real(wp), intent(in) :: val(:) ! matrix values
   integer, intent(in) :: order(:) ! holds pivot order (must be unchanged
      ! since the analyse phase)
   type(MA87_keep), intent(inout) :: keep ! see description of derived type
   type(MA87_control), intent(in) :: control ! see description of derived type
   type(MA87_info) :: info ! see description of derived type

        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MA87_factor with its HSL namesake ', /,                   &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

   info%flag = GALAHAD_unavailable_option
   info%num_factor = 0_long
   info%num_flops = 0_long
   info%num_nodes = 0
   info%maxdepth = 0
   info%stat = 0

end subroutine MA87_factor_double

subroutine MA87_factor_solve_one_double(n, ptr, row, val, order, keep, control,&
      info, x)
   USE GALAHAD_SYMBOLS
   integer, intent(in) :: n ! order of A
   integer, intent(in) :: row(:) ! row indices of lower triangular part
   integer, intent(in) :: ptr(:) ! col pointers for lower triangular part
   real(wp), intent(in) :: val(:) ! matrix values
   integer, intent(in) :: order(:) ! holds pivot order (must be unchanged
      ! since the analyse phase)
   type(MA87_keep), intent(inout) :: keep ! see description of derived type
   type(MA87_control), intent(in) :: control ! see description of derived type
   type(MA87_info) :: info ! see description of derived type
   real(wp), intent(inout) :: x(keep%n) ! On entry, x must
      ! be set so that if i has been used to index a variable,
      ! x(i) is the corresponding component of the right-hand side.
      ! On exit, if i has been used to index a variable,
      ! x(i) holds solution for variable i.

        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MA87_solve_one with its HSL namesake ', /,               &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

   info%flag = GALAHAD_unavailable_option
   info%num_factor = 0_long
   info%num_flops = 0_long
   info%num_nodes = 0
   info%maxdepth = 0
   info%stat = 0
end subroutine MA87_factor_solve_one_double

subroutine MA87_factor_solve_mult_double(n, ptr, row, val, order, keep, &
      control, info, nrhs, lx, x)
   USE GALAHAD_SYMBOLS
   integer, intent(in) :: n ! order of A
   integer, intent(in) :: row(:) ! row indices of lower triangular part
   integer, intent(in) :: ptr(:) ! col pointers for lower triangular part
   real(wp), intent(in) :: val(:) ! matrix values
   integer, intent(in) :: order(:) ! holds pivot order (must be unchanged
      ! since the analyse phase)
   type(MA87_keep), intent(inout) :: keep ! see description of derived type
   type(MA87_control), intent(in) :: control ! see description of derived type
   type(MA87_info) :: info ! see description of derived type
   integer, intent(in) :: nrhs ! number of right-hand sides to solver for.
   integer, intent(in) :: lx ! first dimension of x
   real(wp), intent(inout) :: x(lx,nrhs) ! On entry, x must
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
  &     ' subroutine MA87_solve_mult with its HSL namesake ', /,               &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

   info%flag = GALAHAD_unavailable_option
   info%num_factor = 0_long
   info%num_flops = 0_long
   info%num_nodes = 0
   info%maxdepth = 0
   info%stat = 0
end subroutine MA87_factor_solve_mult_double

subroutine MA87_solve_one_double(x,order,keep,control,info,job)
   USE GALAHAD_SYMBOLS
   type(MA87_keep), intent(inout) :: keep
   real(wp), intent(inout) :: x(keep%n) ! On entry, x must
      ! be set so that if i has been used to index a variable,
      ! x(i) is the corresponding component of the right-hand side.
      ! On exit, if i has been used to index a variable,
      ! x(i) holds solution for variable i.
   integer, intent(in) :: order(:) ! pivot order. must be unchanged
   ! For details of keep, control, info : see derived type description
   type(MA87_control), intent(in) :: control
   type(MA87_info) :: info
   integer, optional, intent(in) :: job  ! used to indicate whether
      ! partial solution required
      ! job = 0 or absent: complete solve performed
      ! job = 1 : forward eliminations only (PLx = b)
      ! job = 2 : backsubs only ((PL)^Tx = b)

        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MA87_solve_one with its HSL namesake ', /,                &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

   info%flag = GALAHAD_unavailable_option
   info%num_factor = 0_long
   info%num_flops = 0_long
   info%num_nodes = 0
   info%maxdepth = 0
   info%stat = 0
end subroutine MA87_solve_one_double

subroutine MA87_solve_mult_double(nrhs,lx,x,order,keep, control,info,job)
   USE GALAHAD_SYMBOLS
   integer, intent(in) :: nrhs ! number of right-hand sides to solver for.
   integer, intent(in) :: lx ! first dimension of x
   real(wp), intent(inout) :: x(lx,nrhs) ! On entry, x must
      ! be set so that if i has been used to index a variable,
      ! x(i,j) is the corresponding component of the
      ! right-hand side for the jth system (j = 1,2,..., nrhs).
      ! On exit, if i has been used to index a variable,
      ! x(i,j) holds solution for variable i to system j
   integer, intent(in) :: order(:) ! pivot order. must be unchanged
   ! For details of keep, control, info : see derived type description
   type(MA87_keep), intent(inout) :: keep
   type(MA87_control), intent(in) :: control
   type(MA87_info) :: info
   integer, optional, intent(in) :: job  ! used to indicate whether
      ! partial solution required
      ! job = 0 or absent: complete solve performed
      ! job = 1 : forward eliminations only (PLX = B).
      ! job = 2 : backsubs only ((PL)^TX = B)

        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MA87_solve_mult with its HSL namesake ', /,               &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

   info%flag = GALAHAD_unavailable_option
   info%num_factor = 0_long
   info%num_flops = 0_long
   info%num_nodes = 0
   info%maxdepth = 0
   info%stat = 0
end subroutine MA87_solve_mult_double

subroutine MA87_sparse_fwd_solve_double(nbi,bindex,b,order,invp,               &
                                        nxi,index,x,w,keep,control,info)
   USE GALAHAD_SYMBOLS
   integer, intent(in) :: nbi ! number of nonzero entries in the right-hand side
   integer, intent(in) :: bindex(:) ! First nbi entries must hold  indices of
      !  nonzero entries in the right-hand side.
   real(wp), intent(in) :: b(:) ! If bindex(i)=k, b(k) must hold the k-th
      ! nonzero component of right-hand side; other entries of b not accessed.
   integer, intent(in) :: order(:) ! pivot order. must be unchanged
   integer, intent(in) :: invp(:) ! must hold inverse pivot order so that
      ! invp(j) holds the j-th pivot.
   integer, intent(out) :: nxi ! number of nonzero entries in the solution.
   integer, intent(out) :: index(:) ! First nxi entries holds indices of
      ! nonzero entries in solution.
   real(wp), intent(inout) :: x(:) ! If index(i)=k, x(k) holds the k-th
      ! nonzero component of solution; all other entries of x are set to zero.
   ! For details of keep, control, info : see derived type description
   real(wp), intent(out) :: w(:) ! work array of size n at least n.
   type(MA87_keep), intent(inout) :: keep
   type(MA87_control), intent(in) :: control
   type(MA87_info), intent(out) :: info
        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MA87_solve_mult with its HSL namesake ', /,               &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

   info%flag = GALAHAD_unavailable_option
   info%num_factor = 0_long
   info%num_flops = 0_long
   info%num_nodes = 0
   info%maxdepth = 0
   info%stat = 0
end subroutine MA87_sparse_fwd_solve_double

subroutine MA87_finalise_double(keep, control)
   USE GALAHAD_SYMBOLS
   type(MA87_keep), intent(inout) :: keep    ! See derived-type declaration
   type(MA87_control), intent(in) :: control ! See derived-type declaration

        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MA87_finalize with its HSL namesake ', /,                 &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

end subroutine MA87_finalise_double

subroutine factorize_posdef(a, order, keep, control, info, nrhs, ldr, rhs)
   USE GALAHAD_SYMBOLS
   type(zd11_type), intent(in) :: a ! System matrix in CSC format.
   integer, intent(in) :: order(:)  ! holds pivot order
   type(MA87_keep), intent(inout) :: keep ! see description of derived type
   type(MA87_control), intent(in) :: control ! see description of derived type
   type(MA87_info), intent(inout) :: info ! see description of derived type
   integer, intent(in) :: nrhs  ! number of right-hand sides (maybe = 0)
   integer, intent(in) :: ldr  ! leading extent of rhs
   real(wp), intent(inout) :: rhs(ldr*nrhs)  ! On entry holds rhs data.
      ! Overwritten by partial solution (forward substitution performed).

   ! local derived types
   type(dagtask) :: task ! see description of derived type
   type(taskstack) :: stack ! see description of derived type

   ! local arrays
   real(wp), dimension(:), allocatable :: detlog ! per thread sum of log pivot
   integer, dimension(:), allocatable ::  invp ! used to hold inverse ordering
   integer, dimension(:), allocatable ::  map ! allocated to have size n.
     ! used in copying entries of user's matrix a into factor storage
     ! (keep%fact).
   real(wp), dimension(:,:), allocatable ::  rhs_local ! Local right-hand
     ! side arrays. allocated to have size (nrhs*ldr,0:total_threads)

end subroutine factorize_posdef

subroutine ma87_print_flag(iflag, control, context, st)
   integer, intent(in) :: iflag
   type(ma87_control), intent(in) :: control
   integer, intent(in), optional :: st
   ! context: is an optional assumed size character array of intent(in).
   ! It describes the context under which the error occured
   character (len=*), optional, intent(in) :: context

        IF ( control%unit_error >= 0 ) WRITE( control%unit_error,              &
     "( ' We regret that the solution options that you have ', /,              &
  &     ' chosen are not all freely available with GALAHAD.', /,               &
  &     ' If you have HSL (formerly the Harwell Subroutine', /,                &
  &     ' Library), this option may be enabled by replacing the dummy ', /,    &
  &     ' subroutine MA87_print_flag with its HSL namesake ', /,               &
  &     ' and dependencies. See ', /,                                          &
  &     '   $GALAHAD/src/makedefs/packages for details.' )" )

end subroutine MA87_print_flag

pure integer function ma87_get_n_double(keep)
   type(ma87_keep), intent(in) :: keep
   ma87_get_n_double = keep%n
end function ma87_get_n_double

end module hsl_MA87_double
