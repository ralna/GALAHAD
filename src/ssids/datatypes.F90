! THIS VERSION: GALAHAD 4.1 - 2023-01-25 AT 09:10 GMT.

#include "spral_procedures.h"

!> \file
!> \copyright 2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg
module spral_ssids_types_precision
  use spral_kinds_precision
!$  use omp_lib
  use, intrinsic :: iso_c_binding
  use spral_scaling_precision, only : auction_options, auction_inform
  use GALAHAD_NODEND_precision, only : NODEND_control_type
  implicit none

  private
  public :: smalloc_type, stack_mem_type, stack_type, thread_stats, &
       real_ptr_type, ssids_options, node_type

  real(rp_), parameter, public :: one = 1.0_rp_
  real(rp_), parameter, public :: zero = 0.0_rp_

  integer(ip_), parameter, public :: nemin_default = 32 ! node amalgamation
    ! parameter

  ! Success flag
  integer(ip_), parameter, public :: SSIDS_SUCCESS                 = 0

  ! Error flags
  integer(ip_), parameter, public :: SSIDS_ERROR_CALL_SEQUENCE     = -1
  integer(ip_), parameter, public :: SSIDS_ERROR_A_N_OOR           = -2
  integer(ip_), parameter, public :: SSIDS_ERROR_A_PTR             = -3
  integer(ip_), parameter, public :: SSIDS_ERROR_A_ALL_OOR         = -4
  integer(ip_), parameter, public :: SSIDS_ERROR_SINGULAR          = -5
  integer(ip_), parameter, public :: SSIDS_ERROR_NOT_POS_DEF       = -6
  integer(ip_), parameter, public :: SSIDS_ERROR_PTR_ROW           = -7
  integer(ip_), parameter, public :: SSIDS_ERROR_ORDER             = -8
  integer(ip_), parameter, public :: SSIDS_ERROR_VAL               = -9
  integer(ip_), parameter, public :: SSIDS_ERROR_X_SIZE            = -10
  integer(ip_), parameter, public :: SSIDS_ERROR_JOB_OOR           = -11
  integer(ip_), parameter, public :: SSIDS_ERROR_NOT_LLT           = -13
  integer(ip_), parameter, public :: SSIDS_ERROR_NOT_LDLT          = -14
  integer(ip_), parameter, public :: SSIDS_ERROR_NO_SAVED_SCALING  = -15
  integer(ip_), parameter, public :: SSIDS_ERROR_ALLOCATION        = -50
  integer(ip_), parameter, public :: SSIDS_ERROR_CUDA_UNKNOWN      = -51
  integer(ip_), parameter, public :: SSIDS_ERROR_CUBLAS_UNKNOWN    = -52
!$ integer, parameter, public :: SSIDS_ERROR_OMP_CANCELLATION  = -53
  integer(ip_), parameter, public :: SSIDS_ERROR_NO_METIS          = -97
  integer(ip_), parameter, public :: SSIDS_ERROR_UNIMPLEMENTED     = -98
  integer(ip_), parameter, public :: SSIDS_ERROR_UNKNOWN           = -99

  ! warning flags
  integer(ip_), parameter, public :: SSIDS_WARNING_IDX_OOR          = 1
  integer(ip_), parameter, public :: SSIDS_WARNING_DUP_IDX          = 2
  integer(ip_), parameter, public :: SSIDS_WARNING_DUP_AND_OOR      = 3
  integer(ip_), parameter, public :: SSIDS_WARNING_MISSING_DIAGONAL = 4
  integer(ip_), parameter, public :: SSIDS_WARNING_MISS_DIAG_OORDUP = 5
  integer(ip_), parameter, public :: SSIDS_WARNING_ANAL_SINGULAR    = 6
  integer(ip_), parameter, public :: SSIDS_WARNING_FACT_SINGULAR    = 7
  integer(ip_), parameter, public :: SSIDS_WARNING_MATCH_ORD_NO_SCALE=8
!$ integer, parameter, public :: SSIDS_WARNING_OMP_PROC_BIND    = 50

  ! solve job values
  integer(ip_), parameter, public :: SSIDS_SOLVE_JOB_ALL     = 0 !PLD(PL)^TX = B
  integer(ip_), parameter, public :: SSIDS_SOLVE_JOB_FWD     = 1 !PLX = B
  integer(ip_), parameter, public :: SSIDS_SOLVE_JOB_DIAG    = 2 !DX = B (indef)
  integer(ip_), parameter, public :: SSIDS_SOLVE_JOB_BWD     = 3 !(PL)^TX = B
  integer(ip_), parameter, public :: SSIDS_SOLVE_JOB_DIAG_BWD= 4 !D(PL)^TX=B
                                                                 ! (indef)

  ! NB: the below must match enum PivotMethod in cpu/cpu_iface.hxx
  integer(ip_), parameter, public :: PIVOT_METHOD_APP_AGGRESIVE = 1
  integer(ip_), parameter, public :: PIVOT_METHOD_APP_BLOCK     = 2
  integer(ip_), parameter, public :: PIVOT_METHOD_TPP           = 3

  ! NB: the below must match enum FailedPivotMethod in cpu/cpu_iface.hxx
  integer(ip_), parameter, public :: FAILED_PIVOT_METHOD_TPP    = 1
  integer(ip_), parameter, public :: FAILED_PIVOT_METHOD_PASS   = 2

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! Note: below smalloc etc. types can't be in spral_ssids_alloc module as
  ! they are used as components of later datatypes.

  ! Type for custom allocator
  ! Used to aggregate many small allocations by doing a single big allocation
  ! and chopping it up.
  ! Note: Only supports freeall operation, not individual frees.
  type smalloc_type
     real(rp_), dimension(:), allocatable :: rmem ! real memory
     integer(long_) :: rmem_size ! needed as size(rmem,kind=long) is f2003
     integer(long_) :: rhead = 0 ! last location containing useful information
       ! in rmem
     integer(ip_), dimension(:), allocatable :: imem ! integer memory
     integer(long_) :: imem_size ! needed as size(imem,kind=long) is f2003
     integer(long_) :: ihead = 0 ! last location containing useful information
       ! in imem
     type(smalloc_type), pointer :: next_alloc => null()
     type(smalloc_type), pointer :: top_real => null() ! Last page where real
       ! allocation was successful
     type(smalloc_type), pointer :: top_int => null() ! Last page where integer
       ! allocation was successful
!$     integer(omp_lock_kind) :: lock
  end type smalloc_type

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! Stack memory allocation type
  type stack_mem_type
     real(rp_), dimension(:), allocatable :: mem ! real memory
     integer(long_) :: mem_size ! needed as size(mem,kind=long) is f2003
     integer(long_) :: head = 0 ! last location containing useful information
     type(stack_mem_type), pointer :: below => null() ! next stack frame down
  end type stack_mem_type

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! Data type for storing each node of the factors
  type node_type
     integer(ip_) :: nelim
     integer(ip_) :: ndelay
     integer(long_) :: rdptr ! entry into (rebuilt) rlist_direct
     integer(ip_) :: ncpdb ! #contrib. to parent's diag. block
      type(C_PTR) :: gpu_lcol
      real(rp_), dimension(:), pointer :: lcol ! values in factors
       ! (will also include unneeded data for any columns delayed from this
       ! node)
      integer(ip_), dimension(:), pointer :: perm ! permutation of columns at
       ! this node: perm(i) is column index in expected global elimination
       ! order that is actually eliminated at local elimination index i
       ! Assuming no delays or permutation this will be
       ! sptr(node):sptr(node+1)-1
     ! Following components are used to index directly into contiguous arrays
     ! lcol and perm without taking performance hit for passing pointers
      type(smalloc_type), pointer :: rsmptr, ismptr
     integer(long_) :: rsmsa, ismsa
  end type node_type

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !
  ! Data type for temporary stack data that is only needed transiently during
  ! factorise phase
  ! Each instance represents a "page" of memory
  !
  type stack_type
     real(rp_), dimension(:), pointer :: val => null() ! generated element
     ! Following components allow us to pass contiguous array val without
     ! taking performance hit for passing pointers
     type(stack_mem_type), pointer :: stptr => null()
     integer(long_) :: stsa
  end type stack_type

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !
  ! Data type for per-thread stats. This is amalgamated after end of parallel
  ! section to get info parameters of same name.
  !
  type thread_stats
     integer(ip_) :: flag = SSIDS_SUCCESS
     integer(ip_) :: st = 0
     integer(ip_) :: cuda_error = 0
     integer(ip_) :: cublas_error = 0
     integer(ip_) :: maxfront = 0 ! Maximum front size
     integer(ip_) :: maxsupernode = 0 ! Maximum supernode size
     integer(long_) :: num_factor = 0_long_ ! Number of entries in factors
     integer(long_) :: num_flops = 0_long_ ! Number of floating point operations
     integer(ip_) :: num_delay = 0 ! Number of delayed variables
     integer(ip_) :: num_neg = 0 ! Number of negative pivots
     integer(ip_) :: num_two = 0 ! Number of 2x2 pivots
     integer(ip_) :: num_zero = 0 ! Number of zero pivots
  end type thread_stats

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !
  ! This type is used to pass buf around for each thread such that it can
  ! be reallocated independantly
  !
  type real_ptr_type
     real(rp_), pointer :: chkptr => null()
     real(rp_), dimension(:), allocatable :: val
  end type real_ptr_type

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !
  ! Data type for control parameters
  !
  type ssids_options
     !
     ! Printing options
     !
     integer(ip_) :: print_level = 0 ! Controls diagnostic printing.
       ! Possible values are:
       !  < 0: no printing.
       !  0: error and warning messages only.
       !  1: as 0 plus basic diagnostic printing.
       !  > 1: as 1 plus some more detailed diagnostic messages.
       !  > 9999: debug (absolutely everything - really don't use this)
     integer(ip_) :: unit_diagnostics = 6 ! unit number for diagnostic printing.
       ! Printing is suppressed if unit_diagnostics  <  0.
     integer(ip_) :: unit_error = 6 ! unit number for error messages.
       ! Printing is suppressed if unit_error  <  0.
     integer(ip_) :: unit_warning = 6 ! unit number for warning messages.
       ! Printing is suppressed if unit_warning  <  0.

     !
     ! Options used ssids_analyse() and ssids_analyse_coord()
     !
     integer(ip_) :: ordering = 1 ! controls choice of ordering
       ! 0 Order must be supplied by user
       ! 1 METIS ordering with default settings is used.
       ! 2 Matching with METIS on compressed matrix.
     integer(ip_) :: nemin = nemin_default ! Min. number of eliminations at a
       ! tree node for amalgamation not to be considered.

     !
     ! High level subtree splitting parameters
     !
     logical :: ignore_numa = .true. ! If true, treat entire machine as single
       ! NUMA region for purposes of subtree allocation.
     logical :: use_gpu = .true. ! Use GPUs if present
     logical :: gpu_only = .false. ! FIXME: not yet implemented.
     integer(long_) :: min_gpu_work = 5*10**9_long_ ! Only assign subtree to GPU
       ! if it contains at least this many flops
     real :: max_load_inbalance = 1.2 ! Maximum permissible load inbalance
       ! when dividing tree into subtrees
     real :: gpu_perf_coeff = 1.0 ! How many times better is a GPU than a
       ! single NUMA region's worth of processors

     !
     ! Options used by ssids_factor() [both indef+posdef]
     !
     integer(ip_) :: scaling = 0 ! controls use of scaling.
       !  <=0: user supplied (or no) scaling
       !    1: Matching-based scaling by Hungarian Algorithm (MC64-like)
       !    2: Matching-based scaling by Auction Algorithm
       !    3: Scaling generated during analyse phase for matching-based order
       !  >=4: Norm equilibriation algorithm (MC77-like)

     !
     ! CPU-specific
     !
     integer(long_) :: small_subtree_threshold = 4*10**6 ! Flops below
       ! which we treat a subtree as small and use the single core kernel
     integer(ip_) :: cpu_block_size = 256 ! block size to use for task
       ! generation on larger nodes

     !
     ! Options used by ssids_factor() with posdef=.false.
     !
     logical :: action = .true. ! used in indefinite case only.
       ! If true and the matrix is found to be
       ! singular, computation continues with a warning.
       ! Otherwise, terminates with error SSIDS_ERROR_SINGULAR.
     integer(ip_) :: pivot_method = PIVOT_METHOD_APP_BLOCK
       ! Type of pivoting to use on CPU side:
       ! 0 - A posteori pivoting, roll back entire front on pivot failure
       ! 1 - A posteori pivoting, roll back on block column level for failure
       ! 2 - Traditional threshold partial pivoting (serial, inefficient!)
     real(rp_) :: small = 1e-20_rp_ ! Minimum pivot size (absolute value of a
       ! pivot must be of size at least small to be accepted).
     real(rp_) :: u = 0.01

     type ( NODEND_control_type ) :: nodend_options

     !
     ! Undocumented
     !
     integer(ip_) :: nstream = 1 ! Number of streams to use
     real(rp_) :: multiplier = 1.1 ! size to multiply expected memory size by
       ! when doing initial memory allocation to allow for delays.
     type(auction_options) :: auction ! Auction algorithm parameters
     real :: min_loadbalance = 0.8 ! Minimum load balance required when
       ! finding level set used for multiple streams
     character(len=:), allocatable :: rb_dump ! Filename to dump matrix in
       ! prior to factorization. No dump takes place if not allocated (the
       ! default).
     integer(ip_) :: failed_pivot_method = FAILED_PIVOT_METHOD_TPP
       ! What to do with failed pivots:
       !     <= 1  Attempt to eliminate with TPP pass
       !     >= 2  Pass straight to parent

   contains
     procedure :: print_summary_analyse
     procedure :: print_summary_factor
  end type ssids_options

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  integer(ip_), parameter, public :: BLOCK_SIZE = 8
  integer(ip_), parameter, public :: MNF_BLOCKS = 11
  integer(ip_), parameter, public :: HOGG_ASSEMBLE_TX = 128
  integer(ip_), parameter, public :: HOGG_ASSEMBLE_TY = 8

  integer(ip_), parameter, public :: EXEC_LOC_CPU = 0
  integer(ip_), parameter, public :: EXEC_LOC_GPU = 1

  integer(ip_), parameter, public :: DEBUG_PRINT_LEVEL = 9999

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Print summary of options used in analysis
!> @param this Instance to summarise
!> @param context Name of subroutine to use in printing
  subroutine print_summary_analyse(this, context)
    implicit none
    class(ssids_options), intent(in) :: this
    character(len=*), intent(in) :: context

    integer(ip_) :: mp

    if ((this%print_level .lt. 1) .or. (this%unit_diagnostics .lt. 0)) return
    mp = this%unit_diagnostics
    write (mp,'(/3a)') ' On entry to ', context, ':'
    write (mp,'(a,i15)') ' options%print_level       =  ',this%print_level
    write (mp,'(a,i15)') ' options%unit_diagnostics  =  ',this%unit_diagnostics
    write (mp,'(a,i15)') ' options%unit_error        =  ',this%unit_error
    write (mp,'(a,i15)') ' options%unit_warning      =  ',this%unit_warning
    write (mp,'(a,i15)') ' options%nemin             =  ',this%nemin
    write (mp,'(a,i15)') ' options%ordering          =  ',this%ordering
  end subroutine print_summary_analyse

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Print summary of options used in factorization
!> @param this Instance to summarise
!> @param posdef True if positive-definite factorization to be performed,
!>        false for indefinite.
!> @param context Name of subroutine to use in printing
  subroutine print_summary_factor(this, posdef, context)
    implicit none
    class(ssids_options), intent(in) :: this
    logical, intent(in) :: posdef
    character(len=*), intent(in) :: context

    if ((this%print_level .lt. 1) .or. (this%unit_diagnostics .lt. 0)) return
    if (posdef) then
       write (this%unit_diagnostics,'(//3a,i2,a)') &
            ' Entering ', TRIM(context), ' with posdef = .true. and :'
       write (this%unit_diagnostics,'(a,5(/a,i12),5(/a,es12.4))') &
            ' options parameters (options%) :', &
            ' print_level         Level of diagnostic printing           = ', &
            this%print_level,      &
            ' unit_diagnostics    Unit for diagnostics                   = ', &
            this%unit_diagnostics, &
            ' unit_error          Unit for errors                        = ', &
            this%unit_error,       &
            ' unit_warning        Unit for warnings                      = ', &
            this%unit_warning,     &
            ' scaling             Scaling control                        = ', &
            this%scaling
    else ! indef
       write (this%unit_diagnostics,'(//3a,i2,a)') &
            ' Entering ', TRIM(context), ' with posdef = .false. and :'
       write (this%unit_diagnostics,'(a,5(/a,i12),5(/a,es12.4))') &
            ' options parameters (options%) :', &
            ' print_level         Level of diagnostic printing           = ', &
            this%print_level,      &
            ' unit_diagnostics    Unit for diagnostics                   = ', &
            this%unit_diagnostics, &
            ' unit_error          Unit for errors                        = ', &
            this%unit_error,       &
            ' unit_warning        Unit for warnings                      = ', &
            this%unit_warning,     &
            ' scaling             Scaling control                        = ', &
            this%scaling,          &
            ' small               Small pivot size                       = ', &
            this%small,           &
            ' u                   Initial relative pivot tolerance       = ', &
            this%u,               &
            ' multiplier          Multiplier for increasing array sizes  = ', &
            this%multiplier
    end if
  end subroutine print_summary_factor

end module spral_ssids_types_precision
