! THIS VERSION: GALAHAD 5.4 - 2025-11-29 AT 13:30 GMT

#ifdef REAL_32
#ifdef INTEGER_64
#define GALAHAD_KINDS_precision GALAHAD_KINDS_single_64
#define GALAHAD_SSIDS_precision GALAHAD_SSIDS_single_64
#define GALAHAD_SSIDS_precision_ciface GALAHAD_SSIDS_single_ciface_64
#define GALAHAD_NODEND_precision_ciface GALAHAD_NODEND_single_ciface_64
#else
#define GALAHAD_KINDS_precision GALAHAD_KINDS_single
#define GALAHAD_SSIDS_precision GALAHAD_SSIDS_single
#define GALAHAD_SSIDS_precision_ciface GALAHAD_SSIDS_single_ciface
#define GALAHAD_NODEND_precision_ciface GALAHAD_NODEND_single_ciface
#endif
#elif REAL_128
#ifdef INTEGER_64
#define GALAHAD_KINDS_precision GALAHAD_KINDS_quadruple_64
#define GALAHAD_SSIDS_precision GALAHAD_SSIDS_quadruple_64
#define GALAHAD_SSIDS_precision_ciface GALAHAD_SSIDS_quadruple_ciface_64
#define GALAHAD_NODEND_precision_ciface GALAHAD_NODEND_quadruple_ciface_64
#else
#define GALAHAD_KINDS_precision GALAHAD_KINDS_quadruple
#define GALAHAD_SSIDS_precision GALAHAD_SSIDS_quadruple
#define GALAHAD_SSIDS_precision_ciface GALAHAD_SSIDS_quadruple_ciface
#define GALAHAD_NODEND_precision_ciface GALAHAD_NODEND_quadruple_ciface
#endif
#else
#ifdef INTEGER_64
#define GALAHAD_KINDS_precision GALAHAD_KINDS_double_64
#define GALAHAD_SSIDS_precision GALAHAD_SSIDS_double_64
#define GALAHAD_SSIDS_precision_ciface GALAHAD_SSIDS_double_ciface_64
#define GALAHAD_NODEND_precision_ciface GALAHAD_NODEND_double_ciface_64
#else
#define GALAHAD_KINDS_precision GALAHAD_KINDS_double
#define GALAHAD_SSIDS_precision GALAHAD_SSIDS_double
#define GALAHAD_SSIDS_precision_ciface GALAHAD_SSIDS_double_ciface
#define GALAHAD_NODEND_precision_ciface GALAHAD_NODEND_double_ciface
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

!  C interface module to GALAHAD_SSIDS types and interfaces

  MODULE GALAHAD_SSIDS_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_SSIDS_precision, ONLY : f_ssids_control_type                   &
                                          => SSIDS_control_type,               &
                                        f_ssids_inform_type                    &
                                          => SSIDS_inform_type
    USE GALAHAD_NODEND_precision_ciface, ONLY:                                 &
        nodend_inform_type, nodend_control_type,                               &
        copy_nodend_control_in => copy_control_in,                             &
        copy_nodend_control_out => copy_control_out,                           &
        copy_nodend_inform_out => copy_inform_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: ssids_control_type
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
       TYPE ( nodend_control_type ) :: nodend_control
       INTEGER ( KIND = ipc_ ) :: nstream
       REAL ( KIND = rpc_ ) :: multiplier
!     type(auction_control) :: auction
       REAL ( KIND = rpc_ ) :: min_loadbalance
!    character(len=:), allocatable :: rb_dump
       INTEGER ( KIND = ipc_ ) :: failed_pivot_method
    END TYPE ssids_control_type

    TYPE, BIND( C ) :: ssids_inform_type
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
       TYPE ( nodend_inform_type ) :: nodend_inform
       INTEGER ( KIND = ipc_ ) :: not_first_pass
       INTEGER ( KIND = ipc_ ) :: not_second_pass
       INTEGER ( KIND = ipc_ ) :: nparts
       INTEGER ( KIND = longc_ ) :: cpu_flops
       INTEGER ( KIND = longc_ ) :: gpu_flops
    END TYPE ssids_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, cindexed )
    TYPE ( ssids_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_ssids_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, INTENT( OUT ) :: cindexed

    cindexed = ccontrol%array_base == 0
    fcontrol%print_level = ccontrol%print_level
    fcontrol%unit_diagnostics = ccontrol%unit_diagnostics
    fcontrol%unit_error = ccontrol%unit_error
    fcontrol%unit_warning = ccontrol%unit_warning
    fcontrol%ordering = ccontrol%ordering
    fcontrol%nemin = ccontrol%nemin
    fcontrol%ignore_numa = ccontrol%ignore_numa
    fcontrol%use_gpu = ccontrol%use_gpu
    fcontrol%gpu_only = ccontrol%gpu_only
    fcontrol%min_gpu_work = ccontrol%min_gpu_work
    fcontrol%max_load_inbalance = ccontrol%max_load_inbalance
    fcontrol%gpu_perf_coeff = ccontrol%gpu_perf_coeff
    fcontrol%scaling = ccontrol%scaling
    fcontrol%small_subtree_threshold = ccontrol%small_subtree_threshold
    fcontrol%cpu_block_size = ccontrol%cpu_block_size
    fcontrol%action = ccontrol%action
    fcontrol%pivot_method = ccontrol%pivot_method
    fcontrol%small = ccontrol%small
    fcontrol%u = ccontrol%u
    CALL copy_nodend_control_in( ccontrol%nodend_control,                      &
                                 fcontrol%nodend_control )
    fcontrol%nstream = ccontrol%nstream
    fcontrol%multiplier = ccontrol%multiplier
    fcontrol%min_loadbalance = REAL( ccontrol%min_loadbalance )
    fcontrol%failed_pivot_method = ccontrol%failed_pivot_method
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran information parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_ssids_inform_type ), INTENT( IN ) :: finform
    TYPE ( ssids_inform_type ), INTENT( OUT ) :: cinform

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
    CALL copy_nodend_inform_out( finform%nodend_inform, cinform%nodend_inform )
    cinform%not_first_pass = finform%not_first_pass
    cinform%not_second_pass = finform%not_second_pass
    cinform%nparts = finform%nparts
    cinform%cpu_flops = finform%cpu_flops
    cinform%gpu_flops = finform%gpu_flops
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_SSIDS_precision_ciface
