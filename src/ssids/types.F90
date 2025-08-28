! THIS VERSION: GALAHAD 5.3 - 2025-08-28 AT 13:00 GMT

!-*-*-*-*-*-  G A L A H A D _ S S I D S _ t y p e s   M O D U L E  *-*-*-*-*-

#include "ssids_procedures.h"

!  COPYRIGHT (c) 2016 The Science and Technology Facilities Council (STFC)
!  licence: BSD licence, see LICENCE file for details
!  author: Jonathan Hogg
!  Forked and extended for GALAHAD, Nick Gould, version 3.1, 2016
!  Now incorporating parts of the previous SSIDS_inform

  MODULE GALAHAD_SSIDS_types_precision
    USE GALAHAD_KINDS_precision
!$  USE omp_lib
    USE, INTRINSIC :: iso_c_binding
    USE GALAHAD_MS_precision, ONLY : MS_auction_control_type,                  &
                                     MS_auction_inform_type
    USE GALAHAD_NODEND_precision, ONLY : NODEND_control_type,                  &
                                         NODEND_inform_type
    IMPLICIT NONE

    PRIVATE

!----------------------
!   P a r a m e t e r s
!----------------------

    INTEGER( ip_ ), PARAMETER, PUBLIC :: nemin_default = 32 ! node amalgamation
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

!  data type for information returned by code

    TYPE, PUBLIC :: SSIDS_inform_type

!  takes one of the enumerated flag values:
!    SSIDS_SUCCESS
!    SSIDS_ERROR_XXX
!    SSIDS_WARNING_XXX

       INTEGER( ip_ ) :: flag = SSIDS_SUCCESS
       INTEGER( ip_ ) :: matrix_dup = 0 ! # duplicated entries.
       INTEGER( ip_ ) :: matrix_missing_diag = 0 ! # missing diagonal entries
       INTEGER( ip_ ) :: matrix_outrange = 0 ! # out-of-range entries.
       INTEGER( ip_ ) :: matrix_rank = 0 ! Rank of matrix (anal=structral,
                                       ! fact=actual)
       INTEGER( ip_ ) :: maxdepth = 0 ! Maximum depth of tree
       INTEGER( ip_ ) :: maxfront = 0 ! Maximum front size
       INTEGER( ip_ ) :: maxsupernode = 0 ! Maximum supernode size
       INTEGER( ip_ ) :: num_delay = 0 ! # delayed variables
       INTEGER( long_ ) :: num_factor = 0_long_ ! # entries in factors
       INTEGER( long_ ) :: num_flops = 0_long_ ! # floating point operations
       INTEGER( ip_ ) :: num_neg = 0 ! # negative pivots
       INTEGER( ip_ ) :: num_sup = 0 ! # supernodes
       INTEGER( ip_ ) :: num_two = 0 ! # 2x2 pivots used by factorization
       INTEGER( ip_ ) :: stat = 0 ! stat parameter
       TYPE( MS_auction_inform_type ) :: auction
       INTEGER( ip_ ) :: cuda_error = 0
       INTEGER( ip_ ) :: cublas_error = 0
       TYPE( NODEND_inform_type ) :: nodend_inform

       ! Undocumented FIXME: should we document them?
       INTEGER( ip_ ) :: not_first_pass = 0
       INTEGER( ip_ ) :: not_second_pass = 0
       INTEGER( ip_ ) :: nparts = 0
       INTEGER( long_ ) :: cpu_flops = 0
       INTEGER( long_ ) :: gpu_flops = 0
       ! character( C_CHAR ) :: unused( 76 )
     CONTAINS
       PROCEDURE :: flag_to_character
       PROCEDURE :: print_flag
       PROCEDURE :: reduce
    END TYPE SSIDS_inform_type

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
    RETURN

!  non-executable statements

200 FORMAT( '( //, 3A, I2, A )' )
210 FORMAT( '( // A, 5( / A ,I12  ), 5( / A, ES12.4 ) )' )
    END SUBROUTINE print_summary_factor

!-*-  G A L A H A D -  S S I D S _ flag _ to _ character  F U N C T I O N  -*-

    FUNCTION flag_to_character(this) result( msg )

!  returns a string representation
!  member function inform%flagToCharacter

    IMPLICIT NONE
    CLASS( SSIDS_inform_type ), INTENT( IN ) :: this
    CHARACTER( LEN = 200 ) :: msg ! return value

    SELECT CASE( this%flag )

!  success

    CASE( SSIDS_SUCCESS )
       msg = 'Success'

!  errors
    CASE( SSIDS_ERROR_CALL_SEQUENCE )
       msg = 'Error in sequence of calls.'
    CASE( SSIDS_ERROR_A_N_OOR )
       msg = 'n or ne is out of range (or has changed)'
    CASE( SSIDS_ERROR_A_PTR )
       msg = 'Error in ptr'
    CASE( SSIDS_ERROR_A_ALL_OOR )
       msg = 'All entries in a column out-of-range (ssids_analyse) &
            &or all entries out-of-range (ssids_analyse_coord)'
    CASE( SSIDS_ERROR_SINGULAR )
       msg = 'Matrix found to be singular'
    CASE( SSIDS_ERROR_NOT_POS_DEF )
       msg = 'Matrix is not positive-definite'
    CASE( SSIDS_ERROR_PTR_ROW )
       msg = 'ptr and row should be present'
    CASE( SSIDS_ERROR_ORDER )
       msg = 'Either control%ordering out of range or error in user-supplied  &
            &elimination order'
    CASE( SSIDS_ERROR_X_SIZE )
       msg = 'Error in size of x or nrhs'
    CASE( SSIDS_ERROR_JOB_OOR )
       msg = 'job out of range'
    CASE( SSIDS_ERROR_NOT_LLT )
       msg = 'Not a LL^T factorization of a positive-definite matrix'
    CASE( SSIDS_ERROR_NOT_LDLT )
       msg = 'Not a LDL^T factorization of an indefinite matrix'
    CASE( SSIDS_ERROR_ALLOCATION )
       write ( msg,'( A,I6 )' ) 'Allocation error. stat parameter = ', this%stat
    CASE( SSIDS_ERROR_VAL )
       msg = 'Optional argument val not present when expected'
    CASE( SSIDS_ERROR_NO_SAVED_SCALING )
       msg = 'Requested use of scaling from matching-based &
            &ordering but matching-based ordering not used'
    CASE( SSIDS_ERROR_UNIMPLEMENTED )
       msg = 'Functionality not yet implemented'
!   CASE( SSIDS_ERROR_CUDA_UNKNOWN )
!      WRITE( msg,'( 2A )' ) ' Unhandled CUDA error: ', &
!           trim( cudaGetErrorString( this%cuda_error ) )
    CASE( SSIDS_ERROR_CUBLAS_UNKNOWN )
       msg = 'Unhandled CUBLAS error:'
!$  CASE( SSIDS_ERROR_OMP_CANCELLATION )
!$     msg = 'SSIDS CPU code requires OMP cancellation to be enabled'
    CASE( SSIDS_ERROR_NO_METIS )
       msg = 'MeTiS is not available'

!  warnings

    CASE( SSIDS_WARNING_IDX_OOR )
       msg = 'out-of-range indices detected'
    CASE( SSIDS_WARNING_DUP_IDX )
       msg = 'duplicate entries detected'
    CASE( SSIDS_WARNING_DUP_AND_OOR )
       msg = 'out-of-range indices detected and duplicate entries detected'
    CASE( SSIDS_WARNING_MISSING_DIAGONAL )
       msg = 'one or more diagonal entries is missing'
    CASE( SSIDS_WARNING_MISS_DIAG_OORDUP )
       msg = 'one or more diagonal entries is missing and out-of-range and/or &
            &duplicate entries detected'
    CASE( SSIDS_WARNING_ANALYSIS_SINGULAR )
       msg = 'Matrix found to be structually singular'
    CASE( SSIDS_WARNING_FACT_SINGULAR )
       msg = 'Matrix found to be singular'
    CASE( SSIDS_WARNING_MATCH_ORD_NO_SCALE )
       msg = 'Matching-based ordering used but associated scaling ignored'
!$  CASE( SSIDS_WARNING_OMP_PROC_BIND )
!$     msg = 'OMP_PROC_BIND=false, this may reduce performance'
    CASE DEFAULT
       msg = 'SSIDS Internal Error'
    END SELECT
    RETURN

    END FUNCTION flag_to_character

!-*-  G A L A H A D -  S S I D S _ p r i n t _ f l a g  S U B R O U T I N E  -*-

    SUBROUTINE print_flag( this, control, context )

!  print out warning or error if flag is non-zero
!   this     instance variable
!   control  options to be used for printing
!   context  name of routine to report error from

    IMPLICIT none
    CLASS(  SSIDS_inform_type ), INTENT( IN ) :: this
    TYPE(  SSIDS_control_type ), INTENT( IN ) :: control
    CHARACTER( LEN = * ), INTENT( IN ) :: context

!  local variables

    CHARACTER( LEN = 200 ) :: msg

    IF ( this%flag == SSIDS_SUCCESS ) RETURN ! Nothing to print
    IF ( control%print_level < 0 ) RETURN ! No printing

!  warning

    IF ( this%flag > SSIDS_SUCCESS ) THEN
      IF ( control%unit_warning < 0 ) RETURN ! printing supressed
      WRITE( control%unit_warning,'( / 3A, I0 )' ) ' Warning from ',           &
           TRIM( context ), '. Warning flag = ', this%flag
      msg = this%flag_to_character( )
      WRITE ( control%unit_warning, '( a )' ) msg
    ELSE
      IF ( control%unit_error < 0 ) RETURN ! printing supressed
      WRITE( control%unit_error,'( / 3A, I0 )' ) ' Error return from ',        &
           TRIM( context ), '. Error flag = ', this%flag
      msg = this%flag_to_character( )
      WRITE( control%unit_error, '( A )' ) msg
    END IF
    RETURN

    END SUBROUTINE print_flag

!-*-*-*-  G A L A H A D -  S S I D S _ r e d u c e  S U B R O U T I N E  -*-*-*-

    SUBROUTINE reduce( this, other )

!  combine other's values into this object.
!
!  primarily intended for reducing inform objects after parallel execution.
!   this  instance object
!   other object to reduce values from

    IMPLICIT NONE
    CLASS( SSIDS_inform_type ), INTENT( INOUT ) :: this
    CLASS( SSIDS_inform_type ), INTENT( IN ) :: other

    IF ( this%flag < 0 .OR. other%flag < 0 ) THEN

!  an error is present

      this%flag = MIN( this%flag, other%flag )
    ELSE

!  otherwise only success if both are zero

      this%flag = MAX( this%flag, other%flag )
    END IF
    this%matrix_dup = this%matrix_dup + other%matrix_dup
    this%matrix_missing_diag = this%matrix_missing_diag +                      &
         other%matrix_missing_diag
    this%matrix_outrange = this%matrix_outrange + other%matrix_outrange
    this%matrix_rank = this%matrix_rank + other%matrix_rank
    this%maxdepth = max(this%maxdepth, other%maxdepth)
    this%maxfront = max(this%maxfront, other%maxfront)
    this%maxsupernode = max(this%maxsupernode, other%maxsupernode)
    this%num_delay = this%num_delay + other%num_delay
    this%num_factor = this%num_factor + other%num_factor
    this%num_flops = this%num_flops + other%num_flops
    this%num_neg = this%num_neg + other%num_neg
    this%num_sup = this%num_sup + other%num_sup
    this%num_two = this%num_two + other%num_two
    IF ( other%stat /= 0 ) this%stat = other%stat
! FIXME: %auction ???
    IF ( other%cuda_error /= 0 ) this%cuda_error = other%cuda_error
    IF ( other%cublas_error /= 0 ) this%cublas_error = other%cublas_error
    this%not_first_pass = this%not_first_pass + other%not_first_pass
    this%not_second_pass = this%not_second_pass + other%not_second_pass
    this%nparts = this%nparts + other%nparts
    this%cpu_flops = this%cpu_flops + other%cpu_flops
    this%gpu_flops = this%gpu_flops + other%gpu_flops
    RETURN

    END SUBROUTINE reduce

  END MODULE GALAHAD_SSIDS_types_precision
