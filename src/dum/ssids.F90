! THIS VERSION: GALAHAD 5.3 - 2025-08-31 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-  G A L A H A D  -  D U M M Y   S S I D S   M O D U L E  -*-*-*-*-*-

 MODULE GALAHAD_SSIDS_precision
!$ USE omp_lib
   USE GALAHAD_SYMBOLS
   USE SPRAL_KINDS_precision, ONLY: ip_, ipc_, rp_, long_
   USE, INTRINSIC :: iso_c_binding
   IMPLICIT NONE

   PRIVATE
   PUBLIC :: SSIDS_analyse, SSIDS_analyse_coord, SSIDS_factor,                 &
             SSIDS_solve, SSIDS_free, SSIDS_enquire_posdef,                    &
             SSIDS_enquire_indef, SSIDS_alter

   LOGICAL, PUBLIC, PROTECTED :: ssids_available = .FALSE.

 ! Parameters

   INTEGER( KIND = ip_ ), PARAMETER :: nemin_default = 32
   INTEGER( KIND = ip_ ), PARAMETER, PUBLIC :: PIVOT_METHOD_APP_AGGRESIVE = 1
   INTEGER( KIND = ip_ ), PARAMETER, PUBLIC :: PIVOT_METHOD_APP_BLOCK     = 2
   INTEGER( KIND = ip_ ), PARAMETER, PUBLIC :: PIVOT_METHOD_TPP           = 3

   INTERFACE SSIDS_analyse
     MODULE PROCEDURE analyse_precision
   END INTERFACE SSIDS_analyse

   INTERFACE SSIDS_analyse_coord
     MODULE PROCEDURE SSIDS_analyse_coord_precision
   END INTERFACE SSIDS_analyse_coord

   INTERFACE SSIDS_factor
     MODULE PROCEDURE SSIDS_factor_precision
   END INTERFACE SSIDS_factor

   INTERFACE SSIDS_solve
     MODULE PROCEDURE SSIDS_solve_one_precision
     MODULE PROCEDURE SSIDS_solve_mult_precision
   END INTERFACE SSIDS_solve

   INTERFACE SSIDS_free
     MODULE PROCEDURE free_akeep_precision
     MODULE PROCEDURE free_fkeep_precision
     MODULE PROCEDURE free_both_precision
   END INTERFACE SSIDS_free

   INTERFACE SSIDS_enquire_posdef
     MODULE PROCEDURE SSIDS_enquire_posdef_precision
   END INTERFACE SSIDS_enquire_posdef

   INTERFACE SSIDS_enquire_indef
     MODULE PROCEDURE SSIDS_enquire_indef_precision
   END INTERFACE SSIDS_enquire_indef

   INTERFACE SSIDS_alter
     MODULE PROCEDURE SSIDS_alter_precision
   END INTERFACE SSIDS_alter

! in ../spral/hw_topology.f90

   TYPE :: numa_region
     INTEGER( KIND = ip_ ) :: nproc !< Number of processors in region

!  list of attached GPUs

     INTEGER( KIND = ip_ ), dimension( : ), allocatable :: gpus
   END TYPE numa_region

! in scaling.f90

  TYPE auction_options
      INTEGER( KIND = ip_ ) :: max_iterations = 30000
      INTEGER( KIND = ip_ ) :: max_unchanged( 3 ) = (/ 10, 100, 100 /)
      REAL :: min_proportion( 3 ) = (/ 0.90, 0.0, 0.0 /)
      REAL :: eps_initial = 0.01
   END TYPE auction_options

! in scaling.f90

   TYPE auction_inform
     INTEGER( KIND = ip_ ) :: flag = 0
     INTEGER( KIND = ip_ ) :: stat = 0
     INTEGER( KIND = ip_ ) :: matched = 0
     INTEGER( KIND = ip_ ) :: iterations = 0
     INTEGER( KIND = ip_ ) :: unmatchable = 0
   END TYPE auction_inform

! in subtree.f90

   TYPE, ABSTRACT :: symbolic_subtree_base
     INTEGER( KIND = ip_ ) :: dummy
   END TYPE symbolic_subtree_base

! in subtree.f90

   TYPE, ABSTRACT :: numeric_subtree_base
     INTEGER( KIND = ip_ ) :: dummy
   END TYPE numeric_subtree_base

! in akeep.f90

   TYPE symbolic_subtree_ptr
     INTEGER( KIND = ip_ ) :: exec_loc
!    CLASS( symbolic_subtree_base ), POINTER :: ptr
   END TYPE symbolic_subtree_ptr

! in fkeep.f90

   TYPE numeric_subtree_ptr
     INTEGER( KIND = ip_ ) :: dummy
!    CLASS( numeric_subtree_base ), POINTER :: ptr
   END TYPE numeric_subtree_ptr

! in inform.f90

   TYPE, PUBLIC :: SSIDS_inform_type
     INTEGER( KIND = ip_ ) :: flag = 0
     INTEGER( KIND = ip_ ) :: matrix_dup = 0
     INTEGER( KIND = ip_ ) :: matrix_missing_diag = 0
     INTEGER( KIND = ip_ ) :: matrix_outrange = 0
     INTEGER( KIND = ip_ ) :: matrix_rank = 0
     INTEGER( KIND = ip_ ) :: maxdepth
     INTEGER( KIND = ip_ ) :: maxfront
     INTEGER( KIND = ip_ ) :: num_delay = 0
     INTEGER( KIND = long_ ) :: num_factor = 0_long_
     INTEGER( KIND = long_ ) :: num_flops = 0_long_
     INTEGER( KIND = ip_ ) :: num_neg = 0
     INTEGER( KIND = ip_ ) :: num_sup = 0
     INTEGER( KIND = ip_ ) :: num_two = 0
     INTEGER( KIND = ip_ ) :: stat = 0
     TYPE( auction_inform ) :: auction
     INTEGER( KIND = ip_ ) :: cuda_error
     INTEGER( KIND = ip_ ) :: cublas_error
     INTEGER( KIND = ip_ ) :: not_first_pass = 0
     INTEGER( KIND = ip_ ) :: not_second_pass = 0
     INTEGER( KIND = ip_ ) :: nparts = 0
     INTEGER( KIND = long_ ) :: cpu_flops = 0_long_
     INTEGER( KIND = long_ ) :: gpu_flops = 0_long_
!  CONTAINS
!     PROCEDURE, pass( this ) :: flagToCharacter
   END TYPE SSIDS_inform_type

! in akeep.f90

   TYPE, PUBLIC :: SSIDS_akeep_type
     LOGICAL :: check
     INTEGER( KIND = ip_ ) :: n
     INTEGER( KIND = ip_ ) :: ne
     INTEGER( KIND = ip_ ) :: nnodes = -1
     INTEGER( KIND = ip_ ) :: nparts
     INTEGER( KIND = ip_ ), dimension( : ), allocatable :: part
     TYPE( symbolic_subtree_ptr ), dimension( : ), allocatable :: subtree
     INTEGER( KIND = ip_ ), dimension( : ), allocatable :: contrib_ptr
     INTEGER( KIND = ip_ ), dimension( : ), allocatable :: contrib_idx
     INTEGER( KIND = ipc_ ), dimension( : ), allocatable :: invp
     INTEGER( KIND = ip_ ), dimension( :,: ), allocatable :: nlist
     INTEGER( KIND = ip_ ), dimension( : ), allocatable :: nptr
     INTEGER( KIND = ip_ ), dimension( : ), allocatable :: rlist
     INTEGER( KIND = long_ ), dimension( : ), allocatable :: rptr
     INTEGER( KIND = ip_ ), dimension( : ), allocatable :: sparent
     INTEGER( KIND = ip_ ), dimension( : ), allocatable :: sptr
     INTEGER( KIND = ip_ ), allocatable :: ptr( : )
     INTEGER( KIND = ip_ ), allocatable :: row( : )
     INTEGER( KIND = ip_ ) :: lmap
     INTEGER( KIND = ip_ ), allocatable :: map( : )
     REAL( KIND = rp_ ), dimension( : ), allocatable :: scaling
     TYPE( numa_region ), dimension( : ), allocatable :: topology
     TYPE( SSIDS_inform_type ) :: inform
   END TYPE SSIDS_akeep_type

! in fkeep.f90

   TYPE, PUBLIC :: SSIDS_fkeep_type
     REAL( KIND = rp_ ), dimension( : ), allocatable :: scaling
     LOGICAL :: pos_def
     TYPE( numeric_subtree_ptr ), dimension( : ), allocatable :: subtree
     TYPE( SSIDS_inform_type ) :: inform
   END TYPE SSIDS_fkeep_type

! in datatypes.f90

   TYPE, PUBLIC :: SSIDS_control_type
     INTEGER( KIND = ip_ ) :: print_level = 0
     INTEGER( KIND = ip_ ) :: unit_diagnostics = 6
     INTEGER( KIND = ip_ ) :: unit_error = 6
     INTEGER( KIND = ip_ ) :: unit_warning = 6
     INTEGER( KIND = ip_ ) :: ordering = 1
     INTEGER( KIND = ip_ ) :: nemin = nemin_default
     LOGICAL :: ignore_numa = .true.
     LOGICAL :: use_gpu = .true.
     LOGICAL :: gpu_only = .false.
     INTEGER( KIND = long_ ) :: min_gpu_work = 5*10**9_long_
     REAL :: max_load_inbalance = 1.2
     REAL :: gpu_perf_coeff = 1.0
     INTEGER( KIND = ip_ ) :: scaling = 0
     INTEGER( KIND = long_ ) :: small_subtree_threshold = 4*10**6_long_
     INTEGER( KIND = ip_ ) :: cpu_block_size = 256
     LOGICAL :: action = .true.
     INTEGER( KIND = ip_ ) :: pivot_method = 2
     REAL( KIND = rp_ ) :: small = 1e-20_rp_
     REAL( KIND = rp_ ) :: u = 0.01
     INTEGER( KIND = ip_ ) :: nstream = 1
     REAL( KIND = rp_ ) :: multiplier = 1.1
!    TYPE( auction_options ) :: auction
     REAL :: min_loadbalance = 0.8
!    CHARACTER( LEN = : ), allocatable :: rb_dump
     INTEGER( KIND = ip_ ) :: failed_pivot_method = 1
   END TYPE SSIDS_control_type

 CONTAINS

   SUBROUTINE analyse_precision( check, n, ptr, row, akeep, control, inform,   &
                                 order, val, topology )
   LOGICAL, INTENT( IN ) :: check
   INTEGER( KIND = ip_ ), INTENT( IN ) :: n
   INTEGER( KIND = ip_ ), INTENT( IN ) :: row( : )
   INTEGER( KIND = ip_ ), INTENT( IN ) :: ptr( : )
   TYPE( SSIDS_akeep_type ), INTENT( INOUT ) :: akeep
   TYPE( SSIDS_control_type ), INTENT( IN ) :: control
   TYPE( SSIDS_inform_type ), INTENT( out ) :: inform
   INTEGER( KIND = ip_ ), optional, INTENT( INOUT ) :: order( : )
   REAL( KIND = rp_ ), optional, INTENT( IN ) :: val( : )
   TYPE( numa_region ), dimension( : ), optional, INTENT( IN ) :: topology

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the SSIDS package that you have selected is', /,  &
  &         ' not available with GALAHAD for the compiler you have chosen.' )" )
   inform%flag = GALAHAD_error_unknown_solver

   END SUBROUTINE analyse_precision

   SUBROUTINE SSIDS_analyse_coord_precision( n, ne, row, col, akeep, control,  &
                                             inform, order, val, topology )
   INTEGER( KIND = ip_ ), INTENT( IN ) :: n
   INTEGER( KIND = ip_ ), INTENT( IN ) :: ne
   INTEGER( KIND = ip_ ), INTENT( IN ) :: row( : )
   INTEGER( KIND = ip_ ), INTENT( IN ) :: col( : )
   TYPE( SSIDS_akeep_type ), INTENT( out ) :: akeep
   TYPE( SSIDS_control_type ), INTENT( IN ) :: control
   TYPE( SSIDS_inform_type ), INTENT( out ) :: inform
   INTEGER( KIND = ip_ ), INTENT( INOUT ), optional  :: order( : )
   REAL( KIND = rp_ ), optional, INTENT( IN ) :: val( : )
   TYPE( numa_region ), dimension( : ), optional, INTENT( IN ) :: topology

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the SSIDS package that you have selected is', /,  &
  &         ' not available with GALAHAD for the compiler you have chosen.' )" )
   inform%flag = GALAHAD_error_unknown_solver

   END SUBROUTINE SSIDS_analyse_coord_precision

   SUBROUTINE SSIDS_factor_precision( posdef, val, akeep, fkeep, control,      &
                                      inform, scale, ptr, row )
   logical, INTENT( IN ) :: posdef
   REAL( KIND = rp_ ), dimension( * ), target, INTENT( IN ) :: val
   TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
   TYPE( SSIDS_fkeep_type ), INTENT( INOUT ) :: fkeep
   TYPE( SSIDS_control_type ), INTENT( IN ) :: control
   TYPE( SSIDS_inform_type ), INTENT( out ) :: inform
   REAL( KIND = rp_ ), dimension( : ), optional, INTENT( INOUT ) :: scale
   INTEGER( KIND = ip_ ), dimension( akeep%n+1 ), optional, INTENT( IN ) :: ptr
   INTEGER( KIND = ip_ ), dimension( * ), optional, INTENT( IN ) :: row

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the SSIDS package that you have selected is', /,  &
  &         ' not available with GALAHAD for the compiler you have chosen.' )" )
   inform%flag = GALAHAD_error_unknown_solver

   END SUBROUTINE SSIDS_factor_precision

   SUBROUTINE SSIDS_solve_one_precision( x1, akeep, fkeep, control, inform,    &
                                         job )
   REAL( KIND = rp_ ), dimension( : ), INTENT( INOUT ) :: x1
   TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
   TYPE( SSIDS_fkeep_type ), INTENT( INOUT ) :: fkeep
   TYPE( SSIDS_control_type ), INTENT( IN ) :: control
   TYPE( SSIDS_inform_type ), INTENT( out ) :: inform
   INTEGER( KIND = ip_ ), optional, INTENT( IN ) :: job

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
         "( ' We regret that the SSIDS package that you have selected is', /,  &
  &         ' not available with GALAHAD for the compiler you have chosen.' )" )
   inform%flag = GALAHAD_error_unknown_solver

   END SUBROUTINE SSIDS_solve_one_precision

   SUBROUTINE SSIDS_solve_mult_precision( nrhs, x, ldx, akeep, fkeep, control, &
                                          inform, job )
   INTEGER( KIND = ip_ ), INTENT( IN ) :: nrhs
   INTEGER( KIND = ip_ ), INTENT( IN ) :: ldx
   REAL( KIND = rp_ ), dimension( ldx,nrhs ), INTENT( INOUT ), target :: x
   TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
   TYPE( SSIDS_fkeep_type ), INTENT( INOUT ) :: fkeep
   TYPE( SSIDS_control_type ), INTENT( IN ) :: control
   TYPE( SSIDS_inform_type ), INTENT( out ) :: inform
   INTEGER( KIND = ip_ ), optional, INTENT( IN ) :: job

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the SSIDS package that you have selected is', /,  &
  &         ' not available with GALAHAD for the compiler you have chosen.' )" )
   inform%flag = GALAHAD_error_unknown_solver

   END SUBROUTINE SSIDS_solve_mult_precision

   SUBROUTINE SSIDS_enquire_posdef_precision( akeep, fkeep, control, inform, d )
   TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
   TYPE( SSIDS_fkeep_type ), target, INTENT( IN ) :: fkeep
   TYPE( SSIDS_control_type ), INTENT( IN ) :: control
   TYPE( SSIDS_inform_type ), INTENT( out ) :: inform
   REAL( KIND = rp_ ), dimension( * ), INTENT( out ) :: d

   IF ( control%unit_error >= 0 .AND. control%print_level > 0  )               &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the SSIDS package that you have selected is', /,  &
  &         ' not available with GALAHAD for the compiler you have chosen.' )" )
   inform%flag = GALAHAD_error_unknown_solver

   END SUBROUTINE SSIDS_enquire_posdef_precision

   SUBROUTINE SSIDS_enquire_indef_precision( akeep, fkeep, control, inform,    &
                                             piv_order, d )
   TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
   TYPE( SSIDS_fkeep_type ), target, INTENT( IN ) :: fkeep
   TYPE( SSIDS_control_type ), INTENT( IN ) :: control
   TYPE( SSIDS_inform_type ), INTENT( out ) :: inform
   INTEGER( KIND = ip_ ), dimension( * ), optional, INTENT( out ) :: piv_order
   REAL( KIND = rp_ ), dimension( 2,* ), optional, INTENT( out ) :: d

   IF ( control%unit_error >= 0 .AND. control%print_level > 0  )               &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the SSIDS package that you have selected is', /,  &
  &         ' not available with GALAHAD for the compiler you have chosen.' )" )
   inform%flag = GALAHAD_error_unknown_solver

   END SUBROUTINE SSIDS_enquire_indef_precision

   SUBROUTINE SSIDS_alter_precision( d, akeep, fkeep, control, inform )
   REAL( KIND = rp_ ), dimension( 2, * ), INTENT( IN ) :: d
   TYPE( SSIDS_akeep_type ), INTENT( IN ) :: akeep
   TYPE( SSIDS_fkeep_type ), target, INTENT( INOUT ) :: fkeep
   TYPE( SSIDS_control_type ), INTENT( IN ) :: control
   TYPE( SSIDS_inform_type ), INTENT( out ) :: inform

   IF ( control%unit_error >= 0 .AND. control%print_level > 0 )                &
     WRITE( control%unit_error,                                                &
         "( ' We regret that the SSIDS package that you have selected is', /,  &
  &         ' not available with GALAHAD for the compiler you have chosen.' )" )
   inform%flag = GALAHAD_error_unknown_solver

   END SUBROUTINE SSIDS_alter_precision

   SUBROUTINE free_akeep_precision( akeep, flag )
   TYPE( SSIDS_akeep_type ), INTENT( INOUT ) :: akeep
   INTEGER( KIND = ip_ ), INTENT( out ) :: flag
   flag = GALAHAD_error_unknown_solver
   END SUBROUTINE free_akeep_precision

   SUBROUTINE free_fkeep_precision( fkeep, cuda_error )
   TYPE( SSIDS_fkeep_type ), INTENT( INOUT ) :: fkeep
   INTEGER( KIND = ip_ ), INTENT( out ) :: cuda_error
   cuda_error = GALAHAD_error_unknown_solver
   END SUBROUTINE free_fkeep_precision

   SUBROUTINE free_both_precision( akeep, fkeep, cuda_error )
   TYPE( SSIDS_akeep_type ), INTENT( INOUT ) :: akeep
   TYPE( SSIDS_fkeep_type ), INTENT( INOUT ) :: fkeep
   INTEGER( KIND = ip_ ), INTENT( out ) :: cuda_error
   cuda_error = GALAHAD_error_unknown_solver
   END SUBROUTINE free_both_precision

END MODULE GALAHAD_SSIDS_precision
