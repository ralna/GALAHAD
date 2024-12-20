! THIS VERSION: GALAHAD 5.1 - 2024-12-20 AT 11:25 GMT.

#ifdef REAL_32
#ifdef INTEGER_64
#define SPRAL_KINDS_precision SPRAL_KINDS_single_64
#define SPRAL_SSIDS_precision_ciface SPRAL_SSIDS_single_ciface_64
#define SPRAL_SSIDS_types_precision spral_ssids_types_single_64
#define SPRAL_SSIDS_inform_precision spral_ssids_inform_single_64
#else
#define SPRAL_KINDS_precision SPRAL_KINDS_single
#define SPRAL_SSIDS_precision_ciface SPRAL_SSIDS_single_ciface
#define SPRAL_SSIDS_types_precision spral_ssids_types_single
#define SPRAL_SSIDS_inform_precision spral_ssids_inform_single
#endif
#elif REAL_128
#ifdef INTEGER_64
#define SPRAL_KINDS_precision SPRAL_KINDS_quadruple_64
#define SPRAL_SSIDS_precision_ciface SPRAL_SSIDS_quadruple_ciface_64
#define SPRAL_SSIDS_types_precision spral_ssids_types_quadruple_64
#define SPRAL_SSIDS_inform_precision spral_ssids_inform_quadruple_64
#else
#define SPRAL_KINDS_precision SPRAL_KINDS_quadruple
#define SPRAL_SSIDS_precision_ciface SPRAL_SSIDS_quadruple_ciface
#define SPRAL_SSIDS_types_precision spral_ssids_types_quadruple
#define SPRAL_SSIDS_inform_precision spral_ssids_inform_quadruple
#endif
#else
#ifdef INTEGER_64
#define SPRAL_KINDS_precision SPRAL_KINDS_double_64
#define SPRAL_SSIDS_precision_ciface SPRAL_SSIDS_double_ciface_64
#define SPRAL_SSIDS_types_precision spral_ssids_types_double_64
#define SPRAL_SSIDS_inform_precision spral_ssids_inform_double_64
#else
#define SPRAL_KINDS_precision SPRAL_KINDS_double
#define SPRAL_SSIDS_precision_ciface SPRAL_SSIDS_double_ciface
#define SPRAL_SSIDS_types_precision spral_ssids_types_double
#define SPRAL_SSIDS_inform_precision spral_ssids_inform_double
#endif
#endif

#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _ S S I D S   C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.4. January 3rd 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

!  C interface module to SPRAL_SSIDS types and interfaces

  MODULE SPRAL_SSIDS_precision_ciface
    USE SPRAL_KINDS_precision
    USE SPRAL_SSIDS_types_precision, ONLY : f_ssids_options => ssids_options
    USE SPRAL_SSIDS_inform_precision, ONLY : f_ssids_inform => ssids_inform

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: spral_ssids_options
       INTEGER ( KIND = ipc_ ) :: array_base
       INTEGER ( KIND = ipc_ ) :: print_level
       INTEGER ( KIND = ipc_ ) :: unit_diagnostics
       INTEGER ( KIND = ipc_ ) :: unit_error
       INTEGER ( KIND = ipc_ ) :: unit_warning
       INTEGER ( KIND = ipc_ ) :: ordering
       INTEGER ( KIND = ipc_ ) :: nemin
       LOGICAL ( KIND = C_BOOL ) :: ignore_numa
       LOGICAL ( KIND = C_BOOL ) :: use_gpu
       LOGICAL ( KIND = C_BOOL ) :: gpu_only
       INTEGER ( KIND = longc_ ) :: min_gpu_work
       REAL ( KIND = spc_ ) :: max_load_inbalance
       REAL ( KIND = spc_ ) :: gpu_perf_coeff
       INTEGER ( KIND = ipc_ ) :: scaling
       INTEGER ( KIND = longc_ ) :: small_subtree_threshold
       INTEGER ( KIND = ipc_ ) :: cpu_block_size
       LOGICAL ( KIND = C_BOOL ) :: action
       INTEGER ( KIND = ipc_ ) :: pivot_method
       REAL ( KIND = rpc_ ) :: small
       REAL ( KIND = rpc_ ) :: u
       INTEGER ( KIND = ipc_ ) :: nstream
       REAL ( KIND = rpc_ ) :: multiplier
!     type(auction_options) :: auction
       REAL ( KIND = spc_ ) :: min_loadbalance
!    character(len=:), allocatable :: rb_dump
       INTEGER ( KIND = ipc_ ) :: failed_pivot_method
    END TYPE spral_ssids_options

    TYPE, BIND( C ) :: spral_ssids_inform
       INTEGER ( KIND = ipc_ ) :: flag
       INTEGER ( KIND = ipc_ ) :: matrix_dup
       INTEGER ( KIND = ipc_ ) :: matrix_missing_diag
       INTEGER ( KIND = ipc_ ) :: matrix_outrange
       INTEGER ( KIND = ipc_ ) :: matrix_rank
       INTEGER ( KIND = ipc_ ) :: maxdepth
       INTEGER ( KIND = ipc_ ) :: maxfront
       INTEGER ( KIND = ipc_ ) :: maxsupernode
       INTEGER ( KIND = ipc_ ) :: num_delay
       INTEGER ( KIND = longc_ ) :: num_factor
       INTEGER ( KIND = longc_ ) :: num_flops
       INTEGER ( KIND = ipc_ ) :: num_neg
       INTEGER ( KIND = ipc_ ) :: num_sup
       INTEGER ( KIND = ipc_ ) :: num_two
       INTEGER ( KIND = ipc_ ) :: stat
!    type(auction_inform) :: auction
       INTEGER ( KIND = ipc_ ) :: cuda_error
       INTEGER ( KIND = ipc_ ) :: cublas_error
       INTEGER ( KIND = ipc_ ) :: not_first_pass
       INTEGER ( KIND = ipc_ ) :: not_second_pass
       INTEGER ( KIND = ipc_ ) :: nparts
       INTEGER ( KIND = longc_ ) :: cpu_flops
       INTEGER ( KIND = longc_ ) :: gpu_flops
!      CHARACTER(C_CHAR) :: unused(76)
    END TYPE spral_ssids_inform

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C options parameters to fortran

    SUBROUTINE copy_options_in( coptions, foptions )
    TYPE ( spral_ssids_options ), INTENT( IN ) :: coptions
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
    TYPE ( spral_ssids_inform ), INTENT( OUT ) :: cinform

    cinform%flag = finform%flag
    cinform%matrix_dup = finform%matrix_dup
    cinform%matrix_missing_diag = finform%matrix_missing_diag
    cinform%matrix_outrange = finform%matrix_outrange
    cinform%matrix_rank = finform%matrix_rank
    cinform%maxdepth = finform%maxdepth
    cinform%maxfront = finform%maxfront
    cinform%maxsupernode = finform%maxsupernode
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

  END MODULE SPRAL_SSIDS_precision_ciface
