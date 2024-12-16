! THIS VERSION: GALAHAD 5.1 - 2024-12-15 AT 15:00 GMT.

#include "spral_procedures.h"

!-*-*-*-*-*-  G A L A H A D  -  D U M M Y   S S I D S   M O D U L E  -*-*-*-*-*-

MODULE SPRAL_SSIDS_double
!$ use omp_lib
   USE GALAHAD_SYMBOLS
   USE GALAHAD_KINDS_double
   IMPLICIT NONE

   PRIVATE

   PUBLIC :: ssids_akeep, ssids_fkeep, ssids_options, ssids_inform
   PUBLIC :: SSIDS_analyse, SSIDS_analyse_coord, SSIDS_factor,                 &
             SSIDS_solve, SSIDS_free, SSIDS_enquire_posdef,                    &
             SSIDS_enquire_indef, SSIDS_alter

   LOGICAL, PUBLIC, PARAMETER :: ssids_available = .FALSE.

 ! Parameters

   integer(ip_), parameter :: nemin_default = 32
   integer(ip_), parameter, public :: PIVOT_METHOD_APP_AGGRESIVE = 1
   integer(ip_), parameter, public :: PIVOT_METHOD_APP_BLOCK     = 2
   integer(ip_), parameter, public :: PIVOT_METHOD_TPP           = 3

   interface SSIDS_analyse
      module procedure analyse_double
   end interface SSIDS_analyse

   interface SSIDS_analyse_coord
      module procedure SSIDS_analyse_coord_double
   end interface SSIDS_analyse_coord

   interface SSIDS_factor
      module procedure SSIDS_factor_double
   end interface SSIDS_factor

   interface SSIDS_solve
      module procedure SSIDS_solve_one_double
      module procedure SSIDS_solve_mult_double
   end interface SSIDS_solve

   interface SSIDS_free
      module procedure free_akeep_double
      module procedure free_fkeep_double
      module procedure free_both_double
   end interface SSIDS_free

   interface SSIDS_enquire_posdef
      module procedure SSIDS_enquire_posdef_double
   end interface SSIDS_enquire_posdef

   interface SSIDS_enquire_indef
      module procedure SSIDS_enquire_indef_double
   end interface SSIDS_enquire_indef

   interface SSIDS_alter
      module procedure SSIDS_alter_double
   end interface SSIDS_alter

! in ../spral/hw_topology.f90
   type :: numa_region
      integer(ip_) :: nproc !< Number of processors in region
      integer(ip_), dimension(:), allocatable :: gpus !< List of attached GPUs
   end type numa_region

! in scaling.f90
  type auction_options
      integer(ip_) :: max_iterations = 30000
      integer(ip_) :: max_unchanged(3) = (/ 10, 100, 100 /)
      real :: min_proportion(3) = (/ 0.90, 0.0, 0.0 /)
      real :: eps_initial = 0.01
   end type auction_options

! in scaling.f90
   type auction_inform
      integer(ip_) :: flag = 0
      integer(ip_) :: stat = 0
      integer(ip_) :: matched = 0
      integer(ip_) :: iterations = 0
      integer(ip_) :: unmatchable = 0
   end type auction_inform

! in subtree.f90
   type, abstract :: symbolic_subtree_base
     integer(ip_) :: dummy
   end type symbolic_subtree_base

! in subtree.f90
   type, abstract :: numeric_subtree_base
     integer(ip_) :: dummy
   end type numeric_subtree_base

! in akeep.f90
   type symbolic_subtree_ptr
      integer(ip_) :: exec_loc
!     class(symbolic_subtree_base), pointer :: ptr
   end type symbolic_subtree_ptr

! in fkeep.f90
   type numeric_subtree_ptr
     integer(ip_) :: dummy
!     class(numeric_subtree_base), pointer :: ptr
   end type numeric_subtree_ptr

! in inform.f90
   type ssids_inform
      integer(ip_) :: flag = 0
      integer(ip_) :: matrix_dup = 0
      integer(ip_) :: matrix_missing_diag = 0
      integer(ip_) :: matrix_outrange = 0
      integer(ip_) :: matrix_rank = 0
      integer(ip_) :: maxdepth
      integer(ip_) :: maxfront
      integer(ip_) :: num_delay = 0
      integer(long_) :: num_factor = 0_long_
      integer(long_) :: num_flops = 0_long_
      integer(ip_) :: num_neg = 0
      integer(ip_) :: num_sup = 0
      integer(ip_) :: num_two = 0
      integer(ip_) :: stat = 0
      type(auction_inform) :: auction
      integer(ip_) :: cuda_error
      integer(ip_) :: cublas_error
      integer(ip_) :: not_first_pass = 0
      integer(ip_) :: not_second_pass = 0
      integer(ip_) :: nparts = 0
      integer(long_) :: cpu_flops = 0_long_
      integer(long_) :: gpu_flops = 0_long_
!  contains
!     procedure, pass(this) :: flagToCharacter
   end type ssids_inform

! in akeep.f90
   type ssids_akeep
      logical :: check
      integer(ip_) :: n
      integer(ip_) :: ne
      integer(ip_) :: nnodes = -1
      integer(ip_) :: nparts
      integer(ip_), dimension(:), allocatable :: part
      type(symbolic_subtree_ptr), dimension(:), allocatable :: subtree
      integer(ip_), dimension(:), allocatable :: contrib_ptr
      integer(ip_), dimension(:), allocatable :: contrib_idx
      integer(ipc_), dimension(:), allocatable :: invp
      integer(ip_), dimension(:,:), allocatable :: nlist
      integer(ip_), dimension(:), allocatable :: nptr
      integer(ip_), dimension(:), allocatable :: rlist
      integer(long_), dimension(:), allocatable :: rptr
      integer(ip_), dimension(:), allocatable :: sparent
      integer(ip_), dimension(:), allocatable :: sptr
      integer(ip_), allocatable :: ptr(:)
      integer(ip_), allocatable :: row(:)
      integer(ip_) :: lmap
      integer(ip_), allocatable :: map(:)
      real(dpc_), dimension(:), allocatable :: scaling
      type(numa_region), dimension(:), allocatable :: topology
      type(ssids_inform) :: inform
   end type ssids_akeep

! in fkeep.f90
   type ssids_fkeep
      real(dpc_), dimension(:), allocatable :: scaling
      logical :: pos_def
      type(numeric_subtree_ptr), dimension(:), allocatable :: subtree
      type(ssids_inform) :: inform
   end type ssids_fkeep

! in datatypes.f90
   type SSIDS_options
     integer(ip_) :: print_level = 0
     integer(ip_) :: unit_diagnostics = 6
     integer(ip_) :: unit_error = 6
     integer(ip_) :: unit_warning = 6
     integer(ip_) :: ordering = 1
     integer(ip_) :: nemin = nemin_default
     logical :: ignore_numa = .true.
     logical :: use_gpu = .true.
     logical :: gpu_only = .false.
     integer(long_) :: min_gpu_work = 5*10**9_long_
     real :: max_load_inbalance = 1.2
     real :: gpu_perf_coeff = 1.0
     integer(ip_) :: scaling = 0
     integer(long_) :: small_subtree_threshold = 4*10**6
     integer(ip_) :: cpu_block_size = 256
     logical :: action = .true.
     integer(ip_) :: pivot_method = 2
     real(dpc_) :: small = 1e-20_dpc_
     real(dpc_) :: u = 0.01
     integer(ip_) :: nstream = 1
     real(dpc_) :: multiplier = 1.1
!    type(auction_options) :: auction
     real :: min_loadbalance = 0.8
!    character(len=:), allocatable :: rb_dump
     integer(ip_) :: failed_pivot_method = 1
   end type SSIDS_options

contains

 subroutine analyse_double(check, n, ptr, row, akeep, options, inform,         &
      order, val, topology)
   logical, intent(in) :: check
   integer(ip_), intent(in) :: n
   integer(ip_), intent(in) :: row(:)
   integer(ip_), intent(in) :: ptr(:)
   type(ssids_akeep), intent(inout) :: akeep
   type(SSIDS_options), intent(in) :: options
   type(ssids_inform), intent(out) :: inform
   integer(ip_), optional, intent(inout) :: order(:)
   real(dpc_), optional, intent(in) :: val(:)
   type(numa_region), dimension(:), optional, intent(in) :: topology

   IF ( options%unit_error >= 0 .AND. options%print_level > 0 )                &
     WRITE( options%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have SPRAL, this option may be enabled by', /,            &
  &         ' replacing the dummy subroutine SSIDS_analyse', /,                &
  &         ' with its SPRAL namesake and dependencies. See ', /,              &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   inform%flag = GALAHAD_error_unknown_solver

 end subroutine analyse_double

 subroutine SSIDS_analyse_coord_double(n, ne, row, col, akeep, options, &
      inform, order, val, topology)
   integer(ip_), intent(in) :: n
   integer(ip_), intent(in) :: ne
   integer(ip_), intent(in) :: row(:)
   integer(ip_), intent(in) :: col(:)
   type(SSIDS_akeep), intent(out) :: akeep
   type(ssids_options), intent(in) :: options
   type(SSIDS_inform), intent(out) :: inform
   integer(ip_), intent(inout), optional  :: order(:)
   real(dpc_), optional, intent(in) :: val(:)
   type(numa_region), dimension(:), optional, intent(in) :: topology

   IF ( options%unit_error >= 0 .AND. options%print_level > 0 )                &
     WRITE( options%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have SPRAL, this option may be enabled by', /,            &
  &         ' replacing the dummy subroutine SSIDS_analyse_coord', /,          &
  &         ' with its SPRAL namesake and dependencies. See ', /,              &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   inform%flag = GALAHAD_error_unknown_solver

 end subroutine SSIDS_analyse_coord_double

 subroutine SSIDS_factor_double(posdef, val, akeep, fkeep, options, inform, &
      scale, ptr, row)
   logical, intent(in) :: posdef
   real(dpc_), dimension(*), target, intent(in) :: val
   type(SSIDS_akeep), intent(in) :: akeep
   type(SSIDS_fkeep), intent(inout) :: fkeep
   type(ssids_options), intent(in) :: options
   type(SSIDS_inform), intent(out) :: inform
   real(dpc_), dimension(:), optional, intent(inout) :: scale
   integer(ip_), dimension(akeep%n+1), optional, intent(in) :: ptr
   integer(ip_), dimension(*), optional, intent(in) :: row

   IF ( options%unit_error >= 0 .AND. options%print_level > 0 )                &
     WRITE( options%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have SPRAL, this option may be enabled by', /,            &
  &         ' replacing the dummy subroutine SSIDS_factor', /,                 &
  &         ' with its SPRAL namesake and dependencies. See ', /,              &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   inform%flag = GALAHAD_error_unknown_solver

end subroutine SSIDS_factor_double

 subroutine SSIDS_solve_one_double(x1, akeep, fkeep, options, inform, job)
   real(dpc_), dimension(:), intent(inout) :: x1
   type(SSIDS_akeep), intent(in) :: akeep
   type(SSIDS_fkeep), intent(inout) :: fkeep
   type(ssids_options), intent(in) :: options
   type(SSIDS_inform), intent(out) :: inform
   integer(ip_), optional, intent(in) :: job

   IF ( options%unit_error >= 0 .AND. options%print_level > 0 )                &
     WRITE( options%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have SPRAL, this option may be enabled by', /,            &
  &         ' replacing the dummy subroutine SSIDS_solve_one', /,              &
  &         ' with its SPRAL namesake and dependencies. See ', /,              &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   inform%flag = GALAHAD_error_unknown_solver

 end subroutine SSIDS_solve_one_double

 subroutine SSIDS_solve_mult_double(nrhs, x, ldx, akeep, fkeep, options, &
      inform, job)
   integer(ip_), intent(in) :: nrhs
   integer(ip_), intent(in) :: ldx
   real(dpc_), dimension(ldx,nrhs), intent(inout), target :: x
   type(SSIDS_akeep), intent(in) :: akeep
   type(SSIDS_fkeep), intent(inout) :: fkeep
   type(ssids_options), intent(in) :: options
   type(SSIDS_inform), intent(out) :: inform
   integer(ip_), optional, intent(in) :: job

   IF ( options%unit_error >= 0 .AND. options%print_level > 0 )                &
     WRITE( options%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have SPRAL, this option may be enabled by', /,            &
  &         ' replacing the dummy subroutine SSIDS_solve_mult', /,             &
  &         ' with its SPRAL namesake and dependencies. See ', /,              &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   inform%flag = GALAHAD_error_unknown_solver

 end subroutine SSIDS_solve_mult_double

 subroutine SSIDS_enquire_posdef_double(akeep, fkeep, options, inform, d)
   type(SSIDS_akeep), intent(in) :: akeep
   type(SSIDS_fkeep), target, intent(in) :: fkeep
   type(ssids_options), intent(in) :: options
   type(SSIDS_inform), intent(out) :: inform
   real(dpc_), dimension(*), intent(out) :: d

   IF ( options%unit_error >= 0 .AND. options%print_level > 0 )                &
     WRITE( options%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have SPRAL, this option may be enabled by', /,            &
  &         ' replacing the dummy subroutine SSIDS_enquire_posdef', /,         &
  &         ' with its SPRAL namesake and dependencies. See ', /,              &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   inform%flag = GALAHAD_error_unknown_solver

 end subroutine SSIDS_enquire_posdef_double

 subroutine SSIDS_enquire_indef_double(akeep, fkeep, options, inform, &
      piv_order, d)
   type(SSIDS_akeep), intent(in) :: akeep
   type(SSIDS_fkeep), target, intent(in) :: fkeep
   type(ssids_options), intent(in) :: options
   type(SSIDS_inform), intent(out) :: inform
   integer(ip_), dimension(*), optional, intent(out) :: piv_order
   real(dpc_), dimension(2,*), optional, intent(out) :: d

   IF ( options%unit_error >= 0 .AND. options%print_level > 0 )                &
     WRITE( options%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have SPRAL, this option may be enabled by', /,            &
  &         ' replacing the dummy subroutine SSIDS_enquire_indef', /,          &
  &         ' with its SPRAL namesake and dependencies. See ', /,              &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   inform%flag = GALAHAD_error_unknown_solver

 end subroutine SSIDS_enquire_indef_double

 subroutine SSIDS_alter_double(d, akeep, fkeep, options, inform)
   real(dpc_), dimension(2,*), intent(in) :: d
   type(SSIDS_akeep), intent(in) :: akeep
   type(SSIDS_fkeep), target, intent(inout) :: fkeep
   type(ssids_options), intent(in) :: options
   type(SSIDS_inform), intent(out) :: inform

   IF ( options%unit_error >= 0 .AND. options%print_level > 0 )                &
     WRITE( options%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have SPRAL, this option may be enabled by', /,            &
  &         ' replacing the dummy subroutine SSIDS_alter', /,                  &
  &         ' with its SPRAL namesake and dependencies. See ', /,              &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   inform%flag = GALAHAD_error_unknown_solver

 end subroutine SSIDS_alter_double

 subroutine free_akeep_double(akeep, flag)
   type(SSIDS_akeep), intent(inout) :: akeep
   integer(ip_), intent(out) :: flag
   flag = GALAHAD_error_unknown_solver
 end subroutine free_akeep_double

 subroutine free_fkeep_double(fkeep, cuda_error)
   type(SSIDS_fkeep), intent(inout) :: fkeep
   integer(ip_), intent(out) :: cuda_error
   cuda_error = GALAHAD_error_unknown_solver
 end subroutine free_fkeep_double

 subroutine free_both_double(akeep, fkeep, cuda_error)
   type(SSIDS_akeep), intent(inout) :: akeep
   type(SSIDS_fkeep), intent(inout) :: fkeep
   integer(ip_), intent(out) :: cuda_error
   cuda_error = GALAHAD_error_unknown_solver
 end subroutine free_both_double

END MODULE SPRAL_SSIDS_double
