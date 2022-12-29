! THIS VERSION: 2022-12-29 AT 10:00:00 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M I 2 0   M O D U L E  -*-*-*-

   module hsl_mi20_single

     USE GALAHAD_SYMBOLS
     use hsl_zd11_single
     use hsl_mc65_single
     use hsl_ma48_single

     implicit none

     integer, parameter, private :: myreal = kind(1.0e0)

     type mi20_control
       integer :: aggressive = 1
       integer :: c_fail = 1
       integer :: max_levels = 100
       integer :: max_points = 1
       real (kind=myreal) :: reduction = 0.8
       integer :: st_method = 2
       real(kind=myreal) :: st_parameter = 0.25
       integer :: testing = 1
       real (kind=myreal) :: trunc_parameter = 0.0
       integer :: coarse_solver = 3
       integer :: coarse_solver_its = 10
       real(kind=myreal) :: damping = 0.8
       real(kind=myreal) :: err_tol = 1.0e10
       integer :: levels = -1
       integer :: ma48 = 0
       integer :: pre_smoothing = 2
       integer :: smoother = 2
       integer :: post_smoothing = 2
       integer :: v_iterations = 1
       integer :: print_level = 1
       integer :: print = 6
       integer :: error = 6
       logical :: one_pass_coarsen = .false.
!      real(kind=myreal) ::  tol = -one
!      logical ::  tol_relative = .true.
     end type mi20_control

     type mi20_info
       integer :: flag = 0
       integer :: clevels = 0
       integer :: cpoints = 0
       integer :: cnnz = 0
       integer :: stat
       integer :: getrf_info
       integer :: iterations
       real(kind=myreal) :: residual
       type(ma48_ainfo):: ma48_ainfo
       type(ma48_finfo):: ma48_finfo
       type(ma48_sinfo):: ma48_sinfo
     end type mi20_info

     type mi20_keep
       logical :: new_preconditioner = .true.
       integer :: clevels = 0
       real(kind=myreal), dimension(:,:), allocatable :: lapack_factors
       integer, dimension(:), allocatable :: lapack_pivots
       integer :: st_method
       logical :: lapack_data = .false.
       integer :: ma48_data = 0
       type(ma48_factors) :: ma48_factors
       type(ma48_control) :: ma48_cntrl
       type(zd11_type) :: ma48_matrix
       logical :: ma48_matrix_exists = .false.
       integer :: dsolve_level = -1
       integer :: max_its = 0 
       logical :: zd11_internal_conversion = .false.
       type( zd11_type ) :: A
     end type mi20_keep

     type mi20_data
       type(zd11_type) :: A_mat
       type(zd11_type) :: I_mat
     end type mi20_data

   contains

     subroutine mi20_setup(matrix, coarse_data, keep, control, info)
     type(zd11_type), intent(inout) :: matrix
     type(mi20_data), dimension(:), allocatable, intent(inout) :: coarse_data
     type(mi20_keep), intent(inout) :: keep
     type(mi20_control), intent(in) :: control
     type(mi20_info), intent(out) :: info
     info%flag = GALAHAD_unavailable_option
     return
     end subroutine mi20_setup

     subroutine mi20_finalize(coarse_data, keep, control, info, ma48_cntrl)
     type(mi20_data), allocatable, dimension(:), intent(inout) :: coarse_data
     type(mi20_keep), intent(inout) :: keep
     type(mi20_control), intent(in) ::control
     type(mi20_info), intent(inout) :: info
     type(ma48_control), optional, intent(inout) :: ma48_cntrl
     info%flag = GALAHAD_unavailable_option
     return
     end subroutine mi20_finalize

   end module hsl_mi20_single
