! THIS VERSION: GALAHAD 5.3 - 2025-08-23 AT 14:00 GMT

!-*-*-*-*-*-  G A L A H A D _ S S I D S _ t y p e s   M O D U L E  *-*-*-*-*-

#include "ssids_procedures.h"

!  COPYRIGHT (c) 2016 The Science and Technology Facilities Council (STFC)
!  licence: BSD licence, see LICENCE file for details
!  author: Jonathan Hogg
!  Forked and extended for GALAHAD, Nick Gould, version 3.1, 2016

  MODULE GALAHAD_SSIDS_types_precision
    USE GALAHAD_KINDS_precision
!$  USE omp_lib
    USE, INTRINSIC :: iso_c_binding
    USE GALAHAD_MS_precision, ONLY : MS_auction_control_type
    USE GALAHAD_NODEND_precision, ONLY : NODEND_control_type
    IMPLICIT NONE

    PRIVATE

!----------------------
!   P a r a m e t e r s
!----------------------

    INTEGER( ip_ ), PARAMETER, PUBLIC :: nemin_default = 32 ! node amalgamation
    INTEGER( ip_ ), PARAMETER, PUBLIC :: BLOCK_SIZE = 8
    INTEGER( ip_ ), PARAMETER, PUBLIC :: MNF_BLOCKS = 11
    INTEGER( ip_ ), PARAMETER, PUBLIC :: HOGG_ASSEMBLE_TX = 128
    INTEGER( ip_ ), PARAMETER, PUBLIC :: HOGG_ASSEMBLE_TY = 8
    INTEGER( ip_ ), PARAMETER, PUBLIC :: EXEC_LOC_CPU = 0
    INTEGER( ip_ ), PARAMETER, PUBLIC :: EXEC_LOC_GPU = 1
    INTEGER( ip_ ), PARAMETER, PUBLIC :: DEBUG_PRINT_LEVEL = 9999
    REAL( rp_ ), PARAMETER, PUBLIC :: one = 1.0_rp_
    REAL( rp_ ), PARAMETER, PUBLIC :: zero = 0.0_rp_

!  success flag

    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_SUCCESS                 = 0

!  error flags

    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_CALL_SEQUENCE     = -1
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_A_N_OOR           = -2
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_A_PTR             = -3
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_A_ALL_OOR         = -4
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_SINGULAR          = -5
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_NOT_POS_DEF       = -6
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_PTR_ROW           = -7
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_ORDER             = -8
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_VAL               = -9
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_X_SIZE            = -10
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_JOB_OOR           = -11
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_NOT_LLT           = -13
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_NOT_LDLT          = -14
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_NO_SAVED_SCALING  = -15
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_ALLOCATION        = -50
!   INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_CUDA_UNKNOWN      = -51
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_CUBLAS_UNKNOWN    = -52
!$  INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_OMP_CANCELLATION  = -53
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_NO_METIS          = -97
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_UNIMPLEMENTED     = -98
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_ERROR_UNKNOWN           = -99

!  warning flags

    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_WARNING_IDX_OOR            = 1
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_WARNING_DUP_IDX            = 2
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_WARNING_DUP_AND_OOR        = 3
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_WARNING_MISSING_DIAGONAL   = 4
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_WARNING_MISS_DIAG_OORDUP   = 5
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_WARNING_ANALYSIS_SINGULAR  = 6
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_WARNING_FACT_SINGULAR      = 7
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_WARNING_MATCH_ORD_NO_SCALE = 8
!$  INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_WARNING_OMP_PROC_BIND      = 50

   ! solve job values

    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_SOLVE_JOB_ALL   = 0 !PLD(PL)^TX=B
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_SOLVE_JOB_FWD   = 1 !PLX=B
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_SOLVE_JOB_DIAG  = 2 !DX=B (indef)
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_SOLVE_JOB_BWD   = 3 !(PL)^TX=B
    INTEGER( ip_ ), PARAMETER, PUBLIC :: SSIDS_SOLVE_JOB_DIAG_BWD= 4 !D(PL)^TX=B
                                                                   ! (indef)
! NB: the below must match enum PivotMethod in cpu/cpu_iface.hxx

    INTEGER( ip_ ), PARAMETER, PUBLIC :: PIVOT_METHOD_APP_AGGRESIVE = 1
    INTEGER( ip_ ), PARAMETER, PUBLIC :: PIVOT_METHOD_APP_BLOCK     = 2
    INTEGER( ip_ ), PARAMETER, PUBLIC :: PIVOT_METHOD_TPP           = 3

! NB: the below must match enum FailedPivotMethod in cpu/cpu_iface.hxx

    INTEGER( ip_ ), PARAMETER, PUBLIC :: FAILED_PIVOT_METHOD_TPP    = 1
    INTEGER( ip_ ), PARAMETER, PUBLIC :: FAILED_PIVOT_METHOD_PASS   = 2

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  note: below smalloc etc. types can't be in galahad_ssids_alloc module as
!  they are used as components of later datatypes.

!  type for custom allocator. Used to aggregate many small allocations by 
!  doing a single big allocation and chopping it up.
!  Note: Only supports freeall operation, not individual frees

    TYPE, PUBLIC :: smalloc_type
      REAL( rp_ ), dimension( : ), allocatable :: rmem ! real memory
      INTEGER( long_ ) :: rmem_size ! needed as size(rmem,kind=long) is f2003
      INTEGER( long_ ) :: rhead = 0 ! last location containing useful info

!  in rmem

      INTEGER( ip_ ), dimension( : ), allocatable :: imem ! integer memory
      INTEGER( long_ ) :: imem_size ! needed as size(imem,kind=long) is f2003
      INTEGER( long_ ) :: ihead = 0 ! last location containing useful info

!  in imem

      TYPE( smalloc_type ), pointer :: next_alloc => null()
      TYPE( smalloc_type ), pointer :: top_real => null() ! Last page where
        ! real allocation was successful
      TYPE( smalloc_type ), pointer :: top_int => null() ! Last page where 
        ! integer allocation was successful
!$    INTEGER( omp_lock_kind ) :: lock
    END TYPE smalloc_type

!  stack memory allocation type

    TYPE, PUBLIC :: stack_mem_type
      REAL( rp_ ), DIMENSION( : ), allocatable :: mem ! real memory
      INTEGER( long_ ) :: mem_size ! needed as size( mem,kind=long ) is f2003
      INTEGER( long_ ) :: head = 0 ! last location containing useful info
      TYPE( stack_mem_type ), POINTER :: below => NULL() ! next stack frame down
    END TYPE stack_mem_type

!  data type for storing each node of the factors

    TYPE, PUBLIC :: node_type
      INTEGER( ip_ ) :: nelim
      INTEGER( ip_ ) :: ndelay
      INTEGER( long_ ) :: rdptr ! entry into ( rebuilt ) rlist_direct
      INTEGER( ip_ ) :: ncpdb ! #contrib. to parent's diag. block
      TYPE( C_PTR ) :: gpu_lcol

!  values in factors will also include unneeded data for any columns delayed 
!  from this node

      REAL( rp_ ), DIMENSION( : ), POINTER :: lcol

!  permutation of columns at this node: perm(i) is column index in expected 
!  global elimination order that is actually eliminated at local elimination 
!  index i. Assuming no delays or permutation this will be 
!  sptr(node):sptr(node+1)-1. Following components are used to index directly 
!  into contiguous arrays lcol and perm without taking performance hit 
!  for passing pointers

      INTEGER( ip_ ), DIMENSION( : ), pointer :: perm
      TYPE( smalloc_type ), POINTER :: rsmptr, ismptr
      INTEGER( long_ ) :: rsmsa, ismsa
    END TYPE node_type

!  data type for temporary stack data that is only needed transiently during
!  factorise phase. Each instance represents a "page" of memory

    TYPE, PUBLIC :: stack_type
      REAL( rp_ ), DIMENSION( : ), POINTER :: val => NULL( ) ! generated element
      ! Following components allow us to pass contiguous array val without
      ! taking performance hit for passing pointers
      TYPE( stack_mem_type ), POINTER :: stptr => NULL( )
      INTEGER( long_ ) :: stsa
    END TYPE stack_type

!  data type for per-thread stats. This is amalgamated after end of parallel
!  section to get info parameters of same name.

    TYPE, PUBLIC :: thread_stats
      INTEGER( ip_ ) :: flag = SSIDS_SUCCESS
      INTEGER( ip_ ) :: st = 0
      INTEGER( ip_ ) :: cuda_error = 0
      INTEGER( ip_ ) :: cublas_error = 0
      INTEGER( ip_ ) :: maxfront = 0 ! Maximum front size
      INTEGER( ip_ ) :: maxsupernode = 0 ! Maximum supernode size
      INTEGER( long_ ) :: num_factor = 0_long_ ! # entries in factors
      INTEGER( long_ ) :: num_flops = 0_long_ ! # floating point operations
      INTEGER( ip_ ) :: num_delay = 0 ! # delayed variables
      INTEGER( ip_ ) :: num_neg = 0 ! # negative pivots
      INTEGER( ip_ ) :: num_two = 0 ! # 2x2 pivots
      INTEGER( ip_ ) :: num_zero = 0 ! # zero pivots
    END TYPE thread_stats

!  this type is used to pass buf around for each thread such that it can
!  be reallocated independantly

    TYPE, PUBLIC :: real_ptr_type
      REAL( rp_ ), POINTER :: chkptr => NULL( )
      REAL( rp_ ), DIMENSION( : ), ALLOCATABLE :: val
    END TYPE real_ptr_type

!  data type for control parameters

    TYPE, PUBLIC :: SSIDS_control_type

!  printing options

!  Controls diagnostic printing. Possible values are:
!   < 0: no printing.
!   0: error and warning messages only.
!   1: as 0 plus basic diagnostic printing.
!   > 1: as 1 plus some more detailed diagnostic messages.
!   > 9999: debug (absolutely everything - really don't use this)

      INTEGER( ip_ ) :: print_level = 0 

!  unit number for diagnostic printing. No printing if unit_diagnostics < 0

      INTEGER( ip_ ) :: unit_diagnostics = 6

!  unit number for error messages. No printing if unit_error < 0

      INTEGER( ip_ ) :: unit_error = 6

!  unit number for warning messages. No printing if unit_warning < 0

      INTEGER( ip_ ) :: unit_warning = 6 

!  options used ssids_analyse() and ssids_analyse_coord()

!  controls choice of ordering
!   0 Order must be supplied by user
!   1 METIS ordering with default settings is used.
!   2 Matching with METIS on compressed matrix.

      INTEGER( ip_ ) :: ordering = 1

!  minimum number of eliminations at a tree node for amalgamation not 
!  to be considered

      INTEGER( ip_ ) :: nemin = nemin_default 

!  high level subtree splitting parameters

!  If true, treat entire machine as single NUMA region for purposes of 
!  subtree allocation.

      LOGICAL :: ignore_numa = .TRUE.

      LOGICAL :: use_gpu = .TRUE. ! Use GPUs if present
      LOGICAL :: gpu_only = .FALSE. ! FIXME: not yet implemented.

!  only assign subtree to GPU if it contains at least this many flops

      INTEGER( long_ ) :: min_gpu_work = 5*10**9_long_

!  Maximum permissible load inbalance when dividing tree into subtrees

      REAL :: max_load_inbalance = 1.2

!  How many times better is GPU than a single NUMA region's worth of processors

      REAL :: gpu_perf_coeff = 1.0

!  options used by ssids_factor() [both indef+posdef]

!  controls use of scaling.
!   <=0: user supplied ( or no ) scaling
!     1: Matching-based scaling by Hungarian Algorithm (MC64-like)
!     2: Matching-based scaling by Auction Algorithm
!     3: Scaling generated during analyse phase for matching-based order
!   >=4: Norm equilibriation algorithm (MC77-like)

      INTEGER( ip_ ) :: scaling = 0 

!  CPU-specific

!  flops below which we treat a subtree as small and use the single core kernel

      INTEGER( long_ ) :: small_subtree_threshold = 4*10**6 

!  block size to use for task generation on larger nodes

      INTEGER( ip_ ) :: cpu_block_size = 256

!  options used by ssids_factor() with posdef=.false.

!  used in indefinite case only. If true and the matrix is found to be
!  singular, computation continues with a warning. Otherwise, terminates 
!  with error SSIDS_ERROR_SINGULAR

      LOGICAL :: action = .TRUE.

!  type of pivoting to use on CPU side:
!   0 - A posteori pivoting, roll back entire front on pivot failure
!   1 - A posteori pivoting, roll back on block column level for failure
!   2 - Traditional threshold partial pivoting (serial, inefficient!)

      INTEGER( ip_ ) :: pivot_method = PIVOT_METHOD_APP_BLOCK

!  minimum pivot size (absolute value of a pivot must be of size at least 
!  small to be accepted).

      REAL( rp_ ) :: small = 1e-20_rp_
      REAL( rp_ ) :: u = 0.01

!  options to control node nested dissection ordering

      TYPE ( NODEND_control_type ) :: nodend_control

!  ------------
!  undocumented
!  ------------

!  number of streams to use

      INTEGER( ip_ ) :: nstream = 1

! size to multiply expected memory size by when doing initial memory 
!  allocation to allow for delays

      REAL( rp_ ) :: multiplier = 1.1
      TYPE( MS_auction_control_type ) :: auction ! Auction algorithm parameters

!  minimum load balance required when finding level set use for multiple streams

      REAL :: min_loadbalance = 0.8

!  filename to dump matrix in prior to factorization. No dump takes place 
!  if not allocated (the default)

      CHARACTER( LEN = : ), ALLOCATABLE :: rb_dump

!  what to do with failed pivots:
!     <= 1  Attempt to eliminate with TPP pass
!     >= 2  Pass straight to parent

      INTEGER( ip_ ) :: failed_pivot_method = FAILED_PIVOT_METHOD_TPP

    CONTAINS
      PROCEDURE :: print_summary_analyse
      PROCEDURE :: print_summary_factor
    END TYPE SSIDS_control_type

  CONTAINS

!-  G A L A H A D -  S S I D S _ print _ summary _ analyse  S U B R O U T I N E 

    SUBROUTINE print_summary_analyse( this, context )

!  Print summary of options used in analysis
!   this Instance to summarise
!   context Name of subroutine to use in printing

    IMPLICIT none
    CLASS( SSIDS_control_type ), INTENT( IN ) :: this
    CHARACTER( len=* ), INTENT( IN ) :: context

    INTEGER( ip_ ) :: mp

    IF ( this%print_level < 1 .OR. this%unit_diagnostics < 0 ) RETURN
    mp = this%unit_diagnostics
    WRITE( mp,'( / 3A )' ) ' On entry to ', context, ':'
    WRITE( mp, 200 ) ' control%print_level       =  ', this%print_level
    WRITE( mp, 200 ) ' control%unit_diagnostics  =  ', this%unit_diagnostics
    WRITE( mp, 200 ) ' control%unit_error        =  ', this%unit_error
    WRITE( mp, 200 ) ' control%unit_warning      =  ', this%unit_warning
    WRITE( mp, 200 ) ' control%nemin             =  ', this%nemin
    WRITE( mp, 200 ) ' control%ordering          =  ', this%ordering
    RETURN

!  non-executable statement

200 FORMAT( '( A, I15 )' )

    END SUBROUTINE print_summary_analyse

!-  G A L A H A D -  S S I D S _ print _ summary _ factor  S U B R O U T I N E -

    SUBROUTINE print_summary_factor( this, posdef, context )

!  Print summary of options used in factorization
!   this Instance to summarise
!   posdef True if positive-definite factorization to be performed,
!   false for indefinite.
!   context Name of subroutine to use in printing

    IMPLICIT NONE
    CLASS( SSIDS_control_type ), INTENT( IN ) :: this
    LOGICAL, INTENT( IN ) :: posdef
    CHARACTER( LEN = * ), INTENT( IN ) :: context

    IF ( this%print_level < 1 .OR. this%unit_diagnostics < 0 ) RETURN
    IF ( posdef  ) THEN
       WRITE ( this%unit_diagnostics, 200 )                                    &
            ' Entering ', TRIM( context ), ' with posdef = .true. and :'
       WRITE ( this%unit_diagnostics, 210 )                                    &
            ' options parameters (control%) :',                                &
            ' print_level         Level of diagnostic printing           = ',  &
            this%print_level,                                                  &
            ' unit_diagnostics    Unit for diagnostics                   = ',  &
            this%unit_diagnostics,                                             &
            ' unit_error          Unit for errors                        = ',  &
            this%unit_error,                                                   &
            ' unit_warning        Unit for warnings                      = ',  &
            this%unit_warning,                                                 &
            ' scaling             Scaling control                        = ',  &
            this%scaling
    ELSE ! indefinite
       WRITE ( this%unit_diagnostics, 200 )                                    &
            ' Entering ', TRIM( context ), ' with posdef = .false. and :'
       WRITE ( this%unit_diagnostics, 210 )                                    &
            ' control parameters (control%) :',                                &
            ' print_level         Level of diagnostic printing           = ',  &
            this%print_level,                                                  &
            ' unit_diagnostics    Unit for diagnostics                   = ',  &
            this%unit_diagnostics,                                             &
            ' unit_error          Unit for errors                        = ',  &
            this%unit_error,                                                   &
            ' unit_warning        Unit for warnings                      = ',  &
            this%unit_warning,                                                 &
            ' scaling             Scaling control                        = ',  &
            this%scaling,                                                      &
            ' small               Small pivot size                       = ',  &
            this%small,                                                        &
            ' u                   Initial relative pivot tolerance       = ',  &
            this%u,                                                            &
            ' multiplier          Multiplier for increasing array sizes  = ',  &
            this%multiplier
    END IF

!  non-executable statements

200 FORMAT( '( //, 3A, I2, A )' )
210 FORMAT( '( // A, 5( / A ,I12  ), 5( / A, ES12.4 ) )' )
    END SUBROUTINE print_summary_factor

  END MODULE GALAHAD_SSIDS_types_precision
