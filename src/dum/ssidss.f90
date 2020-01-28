! THIS VERSION: 29/07/2016 AT 15:00 GMT.

!-*-*-*-*-*-  G A L A H A D  -  D U M M Y   S S I D S   M O D U L E  -*-*-*-*-*-

MODULE SPRAL_SSIDS
!$ use omp_lib
   USE GALAHAD_SYMBOLS
   USE, intrinsic :: iso_c_binding
   IMPLICIT NONE

   PRIVATE

   PUBLIC :: ssids_akeep, ssids_fkeep, ssids_options, ssids_inform
   PUBLIC :: SSIDS_analyse, SSIDS_analyse_coord, SSIDS_factor,                 &
             SSIDS_solve, SSIDS_free, SSIDS_enquire_posdef,                    &
             SSIDS_enquire_indef, SSIDS_alter

 ! Parameters

   integer, parameter :: wp = C_FLOAT
   integer, parameter :: long = selected_int_kind(18)
   integer, parameter :: nemin_default = 32
   integer, parameter, public :: PIVOT_METHOD_APP_AGGRESIVE = 1
   integer, parameter, public :: PIVOT_METHOD_APP_BLOCK     = 2
   integer, parameter, public :: PIVOT_METHOD_TPP           = 3

   interface SSIDS_analyse
      module procedure analyse_single
   end interface SSIDS_analyse

   interface SSIDS_analyse_coord
      module procedure SSIDS_analyse_coord_single
   end interface SSIDS_analyse_coord

   interface SSIDS_factor
      module procedure SSIDS_factor_single
   end interface SSIDS_factor

   interface SSIDS_solve
      module procedure SSIDS_solve_one_single
      module procedure SSIDS_solve_mult_single
   end interface SSIDS_solve

   interface SSIDS_free
      module procedure free_akeep_single
      module procedure free_fkeep_single
      module procedure free_both_single
   end interface SSIDS_free

   interface SSIDS_enquire_posdef
      module procedure SSIDS_enquire_posdef_single
   end interface SSIDS_enquire_posdef

   interface SSIDS_enquire_indef
      module procedure SSIDS_enquire_indef_single
   end interface SSIDS_enquire_indef

   interface SSIDS_alter
      module procedure SSIDS_alter_single
   end interface SSIDS_alter

! in ../spral/hw_topology.f90
   type :: numa_region
      integer :: nproc !< Number of processors in region
      integer, dimension(:), allocatable :: gpus !< List of attached GPUs
   end type numa_region

! in scaling.f90
  type auction_options
      integer :: max_iterations = 30000
      integer :: max_unchanged(3) = (/ 10, 100, 100 /)
      real :: min_proportion(3) = (/ 0.90, 0.0, 0.0 /)
      real :: eps_initial = 0.01
   end type auction_options

! in scaling.f90
   type auction_inform
      integer :: flag = 0
      integer :: stat = 0
      integer :: matched = 0
      integer :: iterations = 0
      integer :: unmatchable = 0
   end type auction_inform

! in subtree.f90
   type, abstract :: symbolic_subtree_base
     integer :: dummy
   end type symbolic_subtree_base

! in subtree.f90
   type, abstract :: numeric_subtree_base
     integer :: dummy
   end type numeric_subtree_base

! in akeep.f90
   type symbolic_subtree_ptr
      integer :: exec_loc
!     class(symbolic_subtree_base), pointer :: ptr
   end type symbolic_subtree_ptr

! in fkeep.f90
   type numeric_subtree_ptr
     integer :: dummy
!     class(numeric_subtree_base), pointer :: ptr
   end type numeric_subtree_ptr

! in inform.f90
   type ssids_inform
      integer :: flag = 0
      integer :: matrix_dup = 0
      integer :: matrix_missing_diag = 0
      integer :: matrix_outrange = 0
      integer :: matrix_rank = 0
      integer :: maxdepth
      integer :: maxfront
      integer :: num_delay = 0
      integer(long) :: num_factor = 0_long
      integer(long) :: num_flops = 0_long
      integer :: num_neg = 0
      integer :: num_sup = 0
      integer :: num_two = 0
      integer :: stat = 0
      type(auction_inform) :: auction
      integer :: cuda_error
      integer :: cublas_error
      integer :: not_first_pass = 0
      integer :: not_second_pass = 0
      integer :: nparts = 0
      integer(long) :: cpu_flops = 0_long
      integer(long) :: gpu_flops = 0_long
!  contains
!     procedure, pass(this) :: flagToCharacter
   end type ssids_inform

! in akeep.f90
   type ssids_akeep
      logical :: check
      integer :: n
      integer :: ne
      integer :: nnodes = -1
      integer :: nparts
      integer, dimension(:), allocatable :: part
      type(symbolic_subtree_ptr), dimension(:), allocatable :: subtree
      integer, dimension(:), allocatable :: contrib_ptr
      integer, dimension(:), allocatable :: contrib_idx
      integer(C_INT), dimension(:), allocatable :: invp
      integer, dimension(:,:), allocatable :: nlist
      integer, dimension(:), allocatable :: nptr
      integer, dimension(:), allocatable :: rlist
      integer(long), dimension(:), allocatable :: rptr
      integer, dimension(:), allocatable :: sparent
      integer, dimension(:), allocatable :: sptr
      integer, allocatable :: ptr(:)
      integer, allocatable :: row(:)
      integer :: lmap
      integer, allocatable :: map(:)
      real(wp), dimension(:), allocatable :: scaling
      type(numa_region), dimension(:), allocatable :: topology
      type(ssids_inform) :: inform
   end type ssids_akeep

! in fkeep.f90
   type ssids_fkeep
      real(wp), dimension(:), allocatable :: scaling
      logical :: pos_def
      type(numeric_subtree_ptr), dimension(:), allocatable :: subtree
      type(ssids_inform) :: inform
   end type ssids_fkeep

! in ssids.f90
   type SSIDS_options
      integer :: print_level = 0
      integer :: unit_diagnostics = 6
      integer :: unit_error = 6
      integer :: unit_warning = 6
      integer :: ordering = 1
      integer :: nemin = nemin_default
      integer :: scaling = 0
      logical :: action = .true.
      real(wp) :: small = 1e-20_wp
      real(wp) :: u = 0.01
      logical :: use_gpu_factor = .true.
      logical :: use_gpu_solve = .true.
      integer :: nstream = 1
      real(wp) :: multiplier = 1.1
      type(auction_options) :: auction
      real :: min_loadbalance = 0.8
      integer :: cpu_small_subtree_threshold = 4*10**6
      integer :: cpu_task_block_size = 256
      integer(long) :: min_gpu_work = 10**10_long
      real :: max_load_inbalance = 1.5
      real :: gpu_perf_coeff = 1.5
      integer :: pivot_method = PIVOT_METHOD_APP_BLOCK
      logical :: ignore_numa = .true.
      logical :: use_gpu = .true.
!     character(len=:), allocatable :: rb_dump
      character, allocatable, DIMENSION(:) :: rb_dump
   end type SSIDS_options

contains

 subroutine analyse_single(check, n, ptr, row, akeep, options, inform,         &
      order, val, topology)
   logical, intent(in) :: check
   integer, intent(in) :: n
   integer, intent(in) :: row(:)
   integer, intent(in) :: ptr(:)
   type(ssids_akeep), intent(inout) :: akeep
   type(SSIDS_options), intent(in) :: options
   type(ssids_inform), intent(out) :: inform
   integer, optional, intent(inout) :: order(:)
   real(wp), optional, intent(in) :: val(:)
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

 end subroutine analyse_single

 subroutine SSIDS_analyse_coord_single(n, ne, row, col, akeep, options, &
      inform, order, val, topology)
   integer, intent(in) :: n
   integer, intent(in) :: ne
   integer, intent(in) :: row(:)
   integer, intent(in) :: col(:)
   type(SSIDS_akeep), intent(out) :: akeep
   type(ssids_options), intent(in) :: options
   type(SSIDS_inform), intent(out) :: inform
   integer, intent(inout), optional  :: order(:)
   real(wp), optional, intent(in) :: val(:)
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

 end subroutine SSIDS_analyse_coord_single

 subroutine SSIDS_factor_single(posdef, val, akeep, fkeep, options, inform, &
      scale, ptr, row)
   logical, intent(in) :: posdef
   real(wp), dimension(*), target, intent(in) :: val
   type(SSIDS_akeep), intent(in) :: akeep
   type(SSIDS_fkeep), intent(inout) :: fkeep
   type(ssids_options), intent(in) :: options
   type(SSIDS_inform), intent(out) :: inform
   real(wp), dimension(:), optional, intent(inout) :: scale
   integer, dimension(akeep%n+1), optional, intent(in) :: ptr
   integer, dimension(*), optional, intent(in) :: row

   IF ( options%unit_error >= 0 .AND. options%print_level > 0 )                &
     WRITE( options%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have SPRAL, this option may be enabled by', /,            &
  &         ' replacing the dummy subroutine SSIDS_factor', /,                 &
  &         ' with its SPRAL namesake and dependencies. See ', /,              &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   inform%flag = GALAHAD_error_unknown_solver

end subroutine SSIDS_factor_single

 subroutine SSIDS_solve_one_single(x1, akeep, fkeep, options, inform, job)
   real(wp), dimension(:), intent(inout) :: x1
   type(SSIDS_akeep), intent(in) :: akeep
   type(SSIDS_fkeep), intent(inout) :: fkeep
   type(ssids_options), intent(in) :: options
   type(SSIDS_inform), intent(out) :: inform
   integer, optional, intent(in) :: job

   IF ( options%unit_error >= 0 .AND. options%print_level > 0 )                &
     WRITE( options%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have SPRAL, this option may be enabled by', /,            &
  &         ' replacing the dummy subroutine SSIDS_solve_one', /,              &
  &         ' with its SPRAL namesake and dependencies. See ', /,              &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   inform%flag = GALAHAD_error_unknown_solver

 end subroutine SSIDS_solve_one_single

 subroutine SSIDS_solve_mult_single(nrhs, x, ldx, akeep, fkeep, options, &
      inform, job)
   integer, intent(in) :: nrhs
   integer, intent(in) :: ldx
   real(wp), dimension(ldx,nrhs), intent(inout), target :: x
   type(SSIDS_akeep), intent(in) :: akeep
   type(SSIDS_fkeep), intent(inout) :: fkeep
   type(ssids_options), intent(in) :: options
   type(SSIDS_inform), intent(out) :: inform
   integer, optional, intent(in) :: job

   IF ( options%unit_error >= 0 .AND. options%print_level > 0 )                &
     WRITE( options%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have SPRAL, this option may be enabled by', /,            &
  &         ' replacing the dummy subroutine SSIDS_solve_mult', /,             &
  &         ' with its SPRAL namesake and dependencies. See ', /,              &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   inform%flag = GALAHAD_error_unknown_solver

 end subroutine SSIDS_solve_mult_single

 subroutine SSIDS_enquire_posdef_single(akeep, fkeep, options, inform, d)
   type(SSIDS_akeep), intent(in) :: akeep
   type(SSIDS_fkeep), target, intent(in) :: fkeep
   type(ssids_options), intent(in) :: options
   type(SSIDS_inform), intent(out) :: inform
   real(wp), dimension(*), intent(out) :: d

   IF ( options%unit_error >= 0 .AND. options%print_level > 0 )                &
     WRITE( options%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have SPRAL, this option may be enabled by', /,            &
  &         ' replacing the dummy subroutine SSIDS_enquire_posdef', /,         &
  &         ' with its SPRAL namesake and dependencies. See ', /,              &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   inform%flag = GALAHAD_error_unknown_solver

 end subroutine SSIDS_enquire_posdef_single

 subroutine SSIDS_enquire_indef_single(akeep, fkeep, options, inform, &
      piv_order, d)
   type(SSIDS_akeep), intent(in) :: akeep
   type(SSIDS_fkeep), target, intent(in) :: fkeep
   type(ssids_options), intent(in) :: options
   type(SSIDS_inform), intent(out) :: inform
   integer, dimension(*), optional, intent(out) :: piv_order
   real(wp), dimension(2,*), optional, intent(out) :: d

   IF ( options%unit_error >= 0 .AND. options%print_level > 0 )                &
     WRITE( options%unit_error,                                                &
         "( ' We regret that the solution options that you have ', /,          &
  &         ' chosen are not all freely available with GALAHAD.', /,           &
  &         ' If you have SPRAL, this option may be enabled by', /,            &
  &         ' replacing the dummy subroutine SSIDS_enquire_indef', /,          &
  &         ' with its SPRAL namesake and dependencies. See ', /,              &
  &         '   $GALAHAD/src/makedefs/packages for details.' )" )
   inform%flag = GALAHAD_error_unknown_solver

 end subroutine SSIDS_enquire_indef_single

 subroutine SSIDS_alter_single(d, akeep, fkeep, options, inform)
   real(wp), dimension(2,*), intent(in) :: d
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

 end subroutine SSIDS_alter_single

 subroutine free_akeep_single(akeep, flag)
   type(SSIDS_akeep), intent(inout) :: akeep
   integer, intent(out) :: flag
 end subroutine free_akeep_single

 subroutine free_fkeep_single(fkeep, cuda_error)
   type(SSIDS_fkeep), intent(inout) :: fkeep
   integer, intent(out) :: cuda_error
 end subroutine free_fkeep_single

 subroutine free_both_single(akeep, fkeep, cuda_error)
   type(SSIDS_akeep), intent(inout) :: akeep
   type(SSIDS_fkeep), intent(inout) :: fkeep
   integer, intent(out) :: cuda_error
 end subroutine free_both_single

END MODULE SPRAL_SSIDS
