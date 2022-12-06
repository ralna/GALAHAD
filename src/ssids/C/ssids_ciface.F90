! THIS VERSION: GALAHAD 4.1 - 2022-10-19 AT 10:30 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _ S S I D S   C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.4. January 3rd 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

!  C interface module to SPRAL_SSIDS types and interfaces

  MODULE SPRAL_SSIDS_double_ciface
    USE :: iso_c_binding
    USE SPRAL_SSIDS_datatypes, only : f_ssids_options => ssids_options
    USE SPRAL_SSIDS_inform, only : f_ssids_inform => ssids_inform

    IMPLICIT NONE

!--------------------
!   P r e c i s i o n
!--------------------

    INTEGER, PARAMETER :: wp = C_DOUBLE ! double precision
    INTEGER, PARAMETER :: sp = C_FLOAT  ! single precision

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: ssids_options
       INTEGER ( KIND = C_INT ) :: print_level
       INTEGER ( KIND = C_INT ) :: unit_diagnostics
       INTEGER ( KIND = C_INT ) :: unit_error
       INTEGER ( KIND = C_INT ) :: unit_warning
       INTEGER ( KIND = C_INT ) :: ordering
       INTEGER ( KIND = C_INT ) :: nemin
       LOGICAL ( KIND = C_BOOL ) :: ignore_numa
       LOGICAL ( KIND = C_BOOL ) :: use_gpu
       LOGICAL ( KIND = C_BOOL ) :: gpu_only
       INTEGER ( KIND = C_INT64_T ) :: min_gpu_work
       REAL ( KIND = wp ) :: max_load_inbalance
       REAL ( KIND = wp ) :: gpu_perf_coeff
       INTEGER ( KIND = C_INT ) :: scaling
       INTEGER ( KIND = C_INT64_T ) :: small_subtree_threshold
       INTEGER ( KIND = C_INT ) :: cpu_block_size
       LOGICAL ( KIND = C_BOOL ) :: action
       INTEGER ( KIND = C_INT ) :: pivot_method
       REAL ( KIND = wp ) :: small
       REAL ( KIND = wp ) :: u
       INTEGER ( KIND = C_INT ) :: nstream
       REAL ( KIND = wp ) :: multiplier
!     type(auction_options) :: auction 
       REAL ( KIND = wp ) :: min_loadbalance
!    character(len=:), allocatable :: rb_dump 
       INTEGER ( KIND = C_INT ) :: failed_pivot_method
    END TYPE ssids_options

    TYPE, BIND( C ) :: ssids_inform
       INTEGER ( KIND = C_INT ) :: flag
       INTEGER ( KIND = C_INT ) :: matrix_dup
       INTEGER ( KIND = C_INT ) :: matrix_missing_diag
       INTEGER ( KIND = C_INT ) :: matrix_outrange
       INTEGER ( KIND = C_INT ) :: matrix_rank
       INTEGER ( KIND = C_INT ) :: maxdepth
       INTEGER ( KIND = C_INT ) :: maxfront
       INTEGER ( KIND = C_INT ) :: num_delay
       INTEGER ( KIND = C_INT64_T ) :: num_factor
       INTEGER ( KIND = C_INT64_T ) :: num_flops
       INTEGER ( KIND = C_INT ) :: num_neg
       INTEGER ( KIND = C_INT ) :: num_sup
       INTEGER ( KIND = C_INT ) :: num_two
       INTEGER ( KIND = C_INT ) :: stat
!    type(auction_inform) :: auction
       INTEGER ( KIND = C_INT ) :: cuda_error
       INTEGER ( KIND = C_INT ) :: cublas_error
       INTEGER ( KIND = C_INT ) :: not_first_pass
       INTEGER ( KIND = C_INT ) :: not_second_pass
       INTEGER ( KIND = C_INT ) :: nparts
       INTEGER ( KIND = C_INT64_T ) :: cpu_flops
       INTEGER ( KIND = C_INT64_T ) :: gpu_flops
    END TYPE ssids_inform

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C options parameters to fortran

    SUBROUTINE copy_options_in( coptions, foptions )
    TYPE ( ssids_options ), INTENT( IN ) :: coptions
    TYPE ( f_ssids_options ), INTENT( OUT ) :: foptions

    foptions%print_level = coptions%print_level
    foptions%unit_diagnostics = coptions%unit_diagnostics
    foptions%unit_error = coptions%unit_error
    foptions%unit_warning = coptions%unit_warning
    foptions%ordering = coptions%ordering
    foptions%nemin = coptions%nemin
    foptions%ignore_numa = coptions%ignore_numa
    foptions%use_gpu = coptions%use_gpu
    foptions%gpu_only = coptions%gpu_only
    foptions%min_gpu_work = coptions%min_gpu_work
    foptions%max_load_inbalance = coptions%max_load_inbalance
    foptions%gpu_perf_coeff = coptions%gpu_perf_coeff
    foptions%scaling = coptions%scaling
    foptions%small_subtree_threshold = coptions%small_subtree_threshold
    foptions%cpu_block_size = coptions%cpu_block_size
    foptions%action = coptions%action
    foptions%pivot_method = coptions%pivot_method
    foptions%small = coptions%small
    foptions%u = coptions%u
    foptions%nstream = coptions%nstream
    foptions%multiplier = coptions%multiplier
    foptions%min_loadbalance = coptions%min_loadbalance
    foptions%failed_pivot_method = coptions%failed_pivot_method
    RETURN

    END SUBROUTINE copy_options_in

!  copy fortran information parameters to C

    SUBROUTINE copy_inform_out( finform, cinform ) 
    TYPE ( f_ssids_inform ), INTENT( IN ) :: finform
    TYPE ( ssids_inform ), INTENT( OUT ) :: cinform

    cinform%flag = finform%flag
    cinform%matrix_dup = finform%matrix_dup
    cinform%matrix_missing_diag = finform%matrix_missing_diag
    cinform%matrix_outrange = finform%matrix_outrange
    cinform%matrix_rank = finform%matrix_rank
    cinform%maxdepth = finform%maxdepth
    cinform%maxfront = finform%maxfront
    cinform%num_delay = finform%num_delay
    cinform%num_factor = finform%num_factor
    cinform%num_flops = finform%num_flops
    cinform%num_neg = finform%num_neg
    cinform%num_sup = finform%num_sup
    cinform%num_two = finform%num_two
    cinform%stat = finform%stat
    cinform%cuda_error = finform%cuda_error
    cinform%cublas_error = finform%cublas_error
    cinform%not_first_pass = finform%not_first_pass
    cinform%not_second_pass = finform%not_second_pass
    cinform%nparts = finform%nparts
    cinform%cpu_flops = finform%cpu_flops
    cinform%gpu_flops = finform%gpu_flops
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE SPRAL_SSIDS_double_ciface
