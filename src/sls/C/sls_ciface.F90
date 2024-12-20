! THIS VERSION: GALAHAD 5.1 - 2024-12-20 AT 11:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  S L S    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.3. November 27th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_SLS_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_SLS_precision, ONLY:                                           &
        f_sls_control_type => SLS_control_type,                                &
        f_sls_time_type => SLS_time_type,                                      &
        f_sls_inform_type => SLS_inform_type,                                  &
        f_sls_full_data_type => SLS_full_data_type,                            &
        f_sls_initialize => SLS_initialize,                                    &
        f_sls_read_specfile => SLS_read_specfile,                              &
        f_sls_analyse_matrix => SLS_analyse_matrix,                            &
        f_sls_factorize_matrix => SLS_factorize_matrix,                        &
        f_sls_solve_system => SLS_solve_system,                                &
        f_sls_partial_solve => SLS_partial_solve,                              &
        f_sls_reset_control => SLS_reset_control,                              &
        f_sls_information => SLS_information,                                  &
        f_sls_terminate => SLS_terminate

    USE GALAHAD_SILS_precision_ciface, ONLY:                                   &
        sils_control_type,                                                     &
        sils_ainfo_type,                                                       &
        sils_finfo_type,                                                       &
        sils_sinfo_type,                                                       &
        copy_sils_ainfo_in => copy_ainfo_in,                                   &
        copy_sils_finfo_in => copy_finfo_in,                                   &
        copy_sils_sinfo_in => copy_sinfo_in,                                   &
        copy_sils_ainfo_out => copy_ainfo_out,                                 &
        copy_sils_finfo_out => copy_finfo_out,                                 &
        copy_sils_sinfo_out => copy_sinfo_out

    USE hsl_ma57_precision_ciface, ONLY:                                       &
        ma57_control,                                                          &
        ma57_ainfo,                                                            &
        ma57_finfo,                                                            &
        ma57_sinfo,                                                            &
        copy_ma57_ainfo_out => copy_ainfo_out,                                 &
        copy_ma57_finfo_out => copy_finfo_out,                                 &
        copy_ma57_sinfo_out => copy_sinfo_out

    USE hsl_ma77_precision_ciface, ONLY:                                       &
        ma77_info,                                                             &
        ma77_control,                                                          &
        copy_ma77_info_out => copy_info_out,                                   &
        copy_ma77_control_in => copy_control_in

    USE hsl_ma86_precision_ciface, ONLY:                                       &
        ma86_info,                                                             &
        ma86_control,                                                          &
        copy_ma86_info_out => copy_info_out,                                   &
        copy_ma86_control_in => copy_control_in

    USE hsl_ma87_precision_ciface, ONLY:                                       &
        ma87_info,                                                             &
        ma87_control,                                                          &
        copy_ma87_info_out => copy_info_out,                                   &
        copy_ma87_control_in => copy_control_in

    USE hsl_ma97_precision_ciface, ONLY:                                       &
        ma97_info,                                                             &
        ma97_control,                                                          &
        copy_ma97_info_out => copy_info_out,                                   &
        copy_ma97_control_in => copy_control_in

    USE SPRAL_SSIDS_precision_ciface, ONLY:                                    &
        spral_ssids_inform,                                                    &
        spral_ssids_options,                                                   &
        copy_ssids_inform_out => copy_inform_out,                              &
        copy_ssids_options_in => copy_options_in

    USE hsl_mc64_precision_ciface, ONLY:                                       &
        mc64_info,                                                             &
        mc64_control,                                                          &
        copy_mc64_info_out => copy_info_out,                                   &
        copy_mc64_control_in => copy_control_in

    USE hsl_mc68_integer_ciface, ONLY:                                         &
        mc68_info,                                                             &
        mc68_control,                                                          &
        copy_mc68_info_out => copy_info_out,                                   &
        copy_mc68_control_in => copy_control_in

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: sls_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: warning
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: statistics
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: print_level_solver
      INTEGER ( KIND = ipc_ ) :: bits
      INTEGER ( KIND = ipc_ ) :: block_size_kernel
      INTEGER ( KIND = ipc_ ) :: block_size_elimination
      INTEGER ( KIND = ipc_ ) :: blas_block_size_factorize
      INTEGER ( KIND = ipc_ ) :: blas_block_size_solve
      INTEGER ( KIND = ipc_ ) :: node_amalgamation
      INTEGER ( KIND = ipc_ ) :: initial_pool_size
      INTEGER ( KIND = ipc_ ) :: min_real_factor_size
      INTEGER ( KIND = ipc_ ) :: min_integer_factor_size
      INTEGER ( KIND = longc_ ) :: max_real_factor_size
      INTEGER ( KIND = longc_ ) :: max_integer_factor_size
      INTEGER ( KIND = longc_ ) :: max_in_core_store
      REAL ( KIND = rpc_ ) :: array_increase_factor
      REAL ( KIND = rpc_ ) :: array_decrease_factor
      INTEGER ( KIND = ipc_ ) :: pivot_control
      INTEGER ( KIND = ipc_ ) :: ordering
      INTEGER ( KIND = ipc_ ) :: full_row_threshold
      INTEGER ( KIND = ipc_ ) :: row_search_indefinite
      INTEGER ( KIND = ipc_ ) :: scaling
      INTEGER ( KIND = ipc_ ) :: scale_maxit
      REAL ( KIND = rpc_ ) :: scale_thresh
      REAL ( KIND = rpc_ ) :: relative_pivot_tolerance
      REAL ( KIND = rpc_ ) :: minimum_pivot_tolerance
      REAL ( KIND = rpc_ ) :: absolute_pivot_tolerance
      REAL ( KIND = rpc_ ) :: zero_tolerance
      REAL ( KIND = rpc_ ) :: zero_pivot_tolerance
      REAL ( KIND = rpc_ ) :: negative_pivot_tolerance
      REAL ( KIND = rpc_ ) :: static_pivot_tolerance
      REAL ( KIND = rpc_ ) :: static_level_switch
      REAL ( KIND = rpc_ ) :: consistency_tolerance
      INTEGER ( KIND = ipc_ ) :: max_iterative_refinements
      REAL ( KIND = rpc_ ) :: acceptable_residual_relative
      REAL ( KIND = rpc_ ) :: acceptable_residual_absolute
      LOGICAL ( KIND = C_BOOL ) :: multiple_rhs
      LOGICAL ( KIND = C_BOOL ) :: generate_matrix_file
      INTEGER ( KIND = ipc_ ) :: matrix_file_device
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: matrix_file_name
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 401 ) :: out_of_core_directory
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 401 ) ::                         &
         out_of_core_integer_factor_file
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 401 ) ::                         &
         out_of_core_real_factor_file
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 401 ) ::                         &
        out_of_core_real_work_file
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 401 ) ::                         &
         out_of_core_indefinite_file
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 501 ) :: out_of_core_restart_file
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
    END TYPE sls_control_type

    TYPE, BIND( C ) :: sls_time_type
      REAL ( KIND = rpc_ ) :: total
      REAL ( KIND = rpc_ ) :: analyse
      REAL ( KIND = rpc_ ) :: factorize
      REAL ( KIND = rpc_ ) :: solve
      REAL ( KIND = rpc_ ) :: order_external
      REAL ( KIND = rpc_ ) :: analyse_external
      REAL ( KIND = rpc_ ) :: factorize_external
      REAL ( KIND = rpc_ ) :: solve_external
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_analyse
      REAL ( KIND = rpc_ ) :: clock_factorize
      REAL ( KIND = rpc_ ) :: clock_solve
      REAL ( KIND = rpc_ ) :: clock_order_external
      REAL ( KIND = rpc_ ) :: clock_analyse_external
      REAL ( KIND = rpc_ ) :: clock_factorize_external
      REAL ( KIND = rpc_ ) :: clock_solve_external
    END TYPE sls_time_type

    TYPE, BIND( C ) :: sls_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = ipc_ ) :: more_info
      INTEGER ( KIND = ipc_ ) :: entries
      INTEGER ( KIND = ipc_ ) :: out_of_range
      INTEGER ( KIND = ipc_ ) :: duplicates
      INTEGER ( KIND = ipc_ ) :: upper
      INTEGER ( KIND = ipc_ ) :: missing_diagonals
      INTEGER ( KIND = ipc_ ) :: max_depth_assembly_tree
      INTEGER ( KIND = ipc_ ) :: nodes_assembly_tree
      INTEGER ( KIND = longc_ ) :: real_size_desirable
      INTEGER ( KIND = longc_ ) :: integer_size_desirable
      INTEGER ( KIND = longc_ ) :: real_size_necessary
      INTEGER ( KIND = longc_ ) :: integer_size_necessary
      INTEGER ( KIND = longc_ ) :: real_size_factors
      INTEGER ( KIND = longc_ ) :: integer_size_factors
      INTEGER ( KIND = longc_ ) :: entries_in_factors
      INTEGER ( KIND = ipc_ ) :: max_task_pool_size
      INTEGER ( KIND = ipc_ ) :: max_front_size
      INTEGER ( KIND = ipc_ ) :: compresses_real
      INTEGER ( KIND = ipc_ ) :: compresses_integer
      INTEGER ( KIND = ipc_ ) :: two_by_two_pivots
      INTEGER ( KIND = ipc_ ) :: semi_bandwidth
      INTEGER ( KIND = ipc_ ) :: delayed_pivots
      INTEGER ( KIND = ipc_ ) :: pivot_sign_changes
      INTEGER ( KIND = ipc_ ) :: static_pivots
      INTEGER ( KIND = ipc_ ) :: first_modified_pivot
      INTEGER ( KIND = ipc_ ) :: rank
      INTEGER ( KIND = ipc_ ) :: negative_eigenvalues
      INTEGER ( KIND = ipc_ ) :: num_zero
      INTEGER ( KIND = ipc_ ) :: iterative_refinements
      INTEGER ( KIND = longc_ ) :: flops_assembly
      INTEGER ( KIND = longc_ ) :: flops_elimination
      INTEGER ( KIND = longc_ ) :: flops_blas
      REAL ( KIND = rpc_ ) :: largest_modified_pivot
      REAL ( KIND = rpc_ ) :: minimum_scaling_factor
      REAL ( KIND = rpc_ ) :: maximum_scaling_factor
      REAL ( KIND = rpc_ ) :: condition_number_1
      REAL ( KIND = rpc_ ) :: condition_number_2
      REAL ( KIND = rpc_ ) :: backward_error_1
      REAL ( KIND = rpc_ ) :: backward_error_2
      REAL ( KIND = rpc_ ) :: forward_error
      LOGICAL ( KIND = C_BOOL ) :: alternative
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 21 ) :: solver
      TYPE ( sls_time_type ) :: time
      TYPE ( sils_ainfo_type ) :: sils_ainfo
      TYPE ( sils_finfo_type ) :: sils_finfo
      TYPE ( sils_sinfo_type ) :: sils_sinfo
      TYPE ( ma57_ainfo ) :: ma57_ainfo
      TYPE ( ma57_finfo ) :: ma57_finfo
      TYPE ( ma57_sinfo ) :: ma57_sinfo
      TYPE ( ma77_info ) :: ma77_info
      TYPE ( ma86_info ) :: ma86_info
      TYPE ( ma87_info ) :: ma87_info
      TYPE ( ma97_info ) :: ma97_info
      TYPE ( spral_ssids_inform ) :: ssids_inform
      INTEGER ( KIND = ipc_ ), DIMENSION( 10 ) :: mc61_info
      REAL ( KIND = rpc_ ), DIMENSION( 15 ) :: mc61_rinfo
      TYPE ( mc64_info ) :: mc64_info
      TYPE ( mc68_info ) :: mc68_info
      INTEGER ( KIND = ipc_ ), DIMENSION( 10 ) :: mc77_info
      REAL ( KIND = rpc_ ), DIMENSION( 10 ) :: mc77_rinfo
      INTEGER ( KIND = ipc_ ) :: mumps_error
      INTEGER ( KIND = ipc_ ), DIMENSION( 80 ) :: mumps_info
      REAL ( KIND = rpc_ ), DIMENSION( 40 ) :: mumps_rinfo
      INTEGER ( KIND = ipc_ ) :: pardiso_error
      INTEGER ( KIND = ipc_ ), DIMENSION( 64 ) :: pardiso_IPARM
      REAL ( KIND = rpc_ ), DIMENSION( 64 ) :: pardiso_DPARM
      INTEGER ( KIND = ipc_ ) :: mkl_pardiso_error
      INTEGER ( KIND = ipc_ ), DIMENSION( 64 ) :: mkl_pardiso_IPARM
      INTEGER ( KIND = ipc_ ) :: pastix_info
      INTEGER ( KIND = ipc_ ) :: wsmp_error
      INTEGER ( KIND = ipc_ ), DIMENSION( 64 ) :: wsmp_iparm
      REAL ( KIND = rpc_ ), DIMENSION( 64 ) :: wsmp_dparm
      INTEGER ( KIND = ipc_ ) :: mpi_ierr
      INTEGER ( KIND = ipc_ ) :: lapack_error
    END TYPE sls_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( sls_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_sls_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%warning = ccontrol%warning
    fcontrol%out = ccontrol%out
    fcontrol%statistics = ccontrol%statistics
    fcontrol%print_level = ccontrol%print_level
    fcontrol%print_level_solver = ccontrol%print_level_solver
    fcontrol%bits = ccontrol%bits
    fcontrol%block_size_kernel = ccontrol%block_size_kernel
    fcontrol%block_size_elimination = ccontrol%block_size_elimination
    fcontrol%blas_block_size_factorize = ccontrol%blas_block_size_factorize
    fcontrol%blas_block_size_solve = ccontrol%blas_block_size_solve
    fcontrol%node_amalgamation = ccontrol%node_amalgamation
    fcontrol%initial_pool_size = ccontrol%initial_pool_size
    fcontrol%min_real_factor_size = ccontrol%min_real_factor_size
    fcontrol%min_integer_factor_size = ccontrol%min_integer_factor_size
    fcontrol%max_real_factor_size = ccontrol%max_real_factor_size
    fcontrol%max_integer_factor_size = ccontrol%max_integer_factor_size
    fcontrol%max_in_core_store = ccontrol%max_in_core_store
    fcontrol%pivot_control = ccontrol%pivot_control
    fcontrol%ordering = ccontrol%ordering
    fcontrol%full_row_threshold = ccontrol%full_row_threshold
    fcontrol%row_search_indefinite = ccontrol%row_search_indefinite
    fcontrol%scaling = ccontrol%scaling
    fcontrol%scale_maxit = ccontrol%scale_maxit
    fcontrol%max_iterative_refinements = ccontrol%max_iterative_refinements
    fcontrol%matrix_file_device = ccontrol%matrix_file_device

    ! Reals
    fcontrol%array_increase_factor = ccontrol%array_increase_factor
    fcontrol%array_decrease_factor = ccontrol%array_decrease_factor
    fcontrol%scale_thresh = ccontrol%scale_thresh
    fcontrol%relative_pivot_tolerance = ccontrol%relative_pivot_tolerance
    fcontrol%minimum_pivot_tolerance = ccontrol%minimum_pivot_tolerance
    fcontrol%absolute_pivot_tolerance = ccontrol%absolute_pivot_tolerance
    fcontrol%zero_tolerance = ccontrol%zero_tolerance
    fcontrol%zero_pivot_tolerance = ccontrol%zero_pivot_tolerance
    fcontrol%negative_pivot_tolerance = ccontrol%negative_pivot_tolerance
    fcontrol%static_pivot_tolerance = ccontrol%static_pivot_tolerance
    fcontrol%static_level_switch = ccontrol%static_level_switch
    fcontrol%consistency_tolerance = ccontrol%consistency_tolerance
    fcontrol%acceptable_residual_relative                                      &
      = ccontrol%acceptable_residual_relative
    fcontrol%acceptable_residual_absolute                                      &
      = ccontrol%acceptable_residual_absolute

    ! Logicals
    fcontrol%multiple_rhs = ccontrol%multiple_rhs
    fcontrol%generate_matrix_file = ccontrol%generate_matrix_file

    ! Strings
    DO i = 1, LEN( fcontrol%matrix_file_name )
      IF ( ccontrol%matrix_file_name( i ) == C_NULL_CHAR ) EXIT
      fcontrol%matrix_file_name( i : i ) = ccontrol%matrix_file_name( i )
    END DO
    DO i = 1, LEN( fcontrol%out_of_core_directory )
      IF ( ccontrol%out_of_core_directory( i ) == C_NULL_CHAR ) EXIT
      fcontrol%out_of_core_directory( i : i )                                  &
        = ccontrol%out_of_core_directory( i )
    END DO
    DO i = 1, LEN( fcontrol%out_of_core_integer_factor_file )
      IF ( ccontrol%out_of_core_integer_factor_file( i ) == C_NULL_CHAR ) EXIT
      fcontrol%out_of_core_integer_factor_file( i : i )                        &
        = ccontrol%out_of_core_integer_factor_file( i )
    END DO
    DO i = 1, LEN( fcontrol%out_of_core_real_factor_file )
      IF ( ccontrol%out_of_core_real_factor_file( i ) == C_NULL_CHAR ) EXIT
      fcontrol%out_of_core_real_factor_file( i : i )                           &
        = ccontrol%out_of_core_real_factor_file( i )
    END DO
    DO i = 1, LEN( fcontrol%out_of_core_real_work_file )
      IF ( ccontrol%out_of_core_real_work_file( i ) == C_NULL_CHAR ) EXIT
      fcontrol%out_of_core_real_work_file( i : i )                             &
        = ccontrol%out_of_core_real_work_file( i )
    END DO
    DO i = 1, LEN( fcontrol%out_of_core_indefinite_file )
      IF ( ccontrol%out_of_core_indefinite_file( i ) == C_NULL_CHAR ) EXIT
      fcontrol%out_of_core_indefinite_file( i : i )                            &
        = ccontrol%out_of_core_indefinite_file( i )
    END DO
    DO i = 1, LEN( fcontrol%out_of_core_restart_file )
      IF ( ccontrol%out_of_core_restart_file( i ) == C_NULL_CHAR ) EXIT
      fcontrol%out_of_core_restart_file( i : i )                               &
        = ccontrol%out_of_core_restart_file( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_sls_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( sls_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%warning = fcontrol%warning
    ccontrol%out = fcontrol%out
    ccontrol%statistics = fcontrol%statistics
    ccontrol%print_level = fcontrol%print_level
    ccontrol%print_level_solver = fcontrol%print_level_solver
    ccontrol%bits = fcontrol%bits
    ccontrol%block_size_kernel = fcontrol%block_size_kernel
    ccontrol%block_size_elimination = fcontrol%block_size_elimination
    ccontrol%blas_block_size_factorize = fcontrol%blas_block_size_factorize
    ccontrol%blas_block_size_solve = fcontrol%blas_block_size_solve
    ccontrol%node_amalgamation = fcontrol%node_amalgamation
    ccontrol%initial_pool_size = fcontrol%initial_pool_size
    ccontrol%min_real_factor_size = fcontrol%min_real_factor_size
    ccontrol%min_integer_factor_size = fcontrol%min_integer_factor_size
    ccontrol%max_real_factor_size = fcontrol%max_real_factor_size
    ccontrol%max_integer_factor_size = fcontrol%max_integer_factor_size
    ccontrol%max_in_core_store = fcontrol%max_in_core_store
    ccontrol%pivot_control = fcontrol%pivot_control
    ccontrol%ordering = fcontrol%ordering
    ccontrol%full_row_threshold = fcontrol%full_row_threshold
    ccontrol%row_search_indefinite = fcontrol%row_search_indefinite
    ccontrol%scaling = fcontrol%scaling
    ccontrol%scale_maxit = fcontrol%scale_maxit
    ccontrol%max_iterative_refinements = fcontrol%max_iterative_refinements
    ccontrol%matrix_file_device = fcontrol%matrix_file_device

    ! Reals
    ccontrol%array_increase_factor = fcontrol%array_increase_factor
    ccontrol%array_decrease_factor = fcontrol%array_decrease_factor
    ccontrol%scale_thresh = fcontrol%scale_thresh
    ccontrol%relative_pivot_tolerance = fcontrol%relative_pivot_tolerance
    ccontrol%minimum_pivot_tolerance = fcontrol%minimum_pivot_tolerance
    ccontrol%absolute_pivot_tolerance = fcontrol%absolute_pivot_tolerance
    ccontrol%zero_tolerance = fcontrol%zero_tolerance
    ccontrol%zero_pivot_tolerance = fcontrol%zero_pivot_tolerance
    ccontrol%negative_pivot_tolerance = fcontrol%negative_pivot_tolerance
    ccontrol%static_pivot_tolerance = fcontrol%static_pivot_tolerance
    ccontrol%static_level_switch = fcontrol%static_level_switch
    ccontrol%consistency_tolerance = fcontrol%consistency_tolerance
    ccontrol%acceptable_residual_relative                                      &
      = fcontrol%acceptable_residual_relative
    ccontrol%acceptable_residual_absolute                                      &
      = fcontrol%acceptable_residual_absolute

    ! Logicals
    ccontrol%multiple_rhs = fcontrol%multiple_rhs
    ccontrol%generate_matrix_file = fcontrol%generate_matrix_file

    ! Strings
    l = LEN( fcontrol%matrix_file_name )
    DO i = 1, l
      ccontrol%matrix_file_name( i ) = fcontrol%matrix_file_name( i : i )
    END DO
    ccontrol%matrix_file_name( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%out_of_core_directory )
    DO i = 1, l
      ccontrol%out_of_core_directory( i )                                      &
        = fcontrol%out_of_core_directory( i : i )
    END DO
    ccontrol%out_of_core_directory( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%out_of_core_integer_factor_file )
    DO i = 1, l
      ccontrol%out_of_core_integer_factor_file( i )                            &
        = fcontrol%out_of_core_integer_factor_file( i : i )
    END DO
    ccontrol%out_of_core_integer_factor_file( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%out_of_core_real_factor_file )
    DO i = 1, l
      ccontrol%out_of_core_real_factor_file( i )                               &
        = fcontrol%out_of_core_real_factor_file( i : i )
    END DO
    ccontrol%out_of_core_real_factor_file( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%out_of_core_real_work_file )
    DO i = 1, l
      ccontrol%out_of_core_real_work_file( i )                                 &
       = fcontrol%out_of_core_real_work_file( i : i )
    END DO
    ccontrol%out_of_core_real_work_file( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%out_of_core_indefinite_file )
    DO i = 1, l
      ccontrol%out_of_core_indefinite_file( i )                                &
        = fcontrol%out_of_core_indefinite_file( i : i )
    END DO
    ccontrol%out_of_core_indefinite_file( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%out_of_core_restart_file )
    DO i = 1, l
      ccontrol%out_of_core_restart_file( i )                                   &
        = fcontrol%out_of_core_restart_file( i : i )
    END DO
    ccontrol%out_of_core_restart_file( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( sls_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_sls_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%analyse = ctime%analyse
    ftime%factorize = ctime%factorize
    ftime%solve = ctime%solve
    ftime%order_external = ctime%order_external
    ftime%analyse_external = ctime%analyse_external
    ftime%factorize_external = ctime%factorize_external
    ftime%solve_external = ctime%solve_external
    ftime%clock_total = ctime%clock_total
    ftime%clock_analyse = ctime%clock_analyse
    ftime%clock_factorize = ctime%clock_factorize
    ftime%clock_solve = ctime%clock_solve
    ftime%clock_order_external = ctime%clock_order_external
    ftime%clock_analyse_external = ctime%clock_analyse_external
    ftime%clock_factorize_external = ctime%clock_factorize_external
    ftime%clock_solve_external = ctime%clock_solve_external
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_sls_time_type ), INTENT( IN ) :: ftime
    TYPE ( sls_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%analyse = ftime%analyse
    ctime%factorize = ftime%factorize
    ctime%solve = ftime%solve
    ctime%order_external = ftime%order_external
    ctime%analyse_external = ftime%analyse_external
    ctime%factorize_external = ftime%factorize_external
    ctime%solve_external = ftime%solve_external
    ctime%clock_total = ftime%clock_total
    ctime%clock_analyse = ftime%clock_analyse
    ctime%clock_factorize = ftime%clock_factorize
    ctime%clock_solve = ftime%clock_solve
    ctime%clock_order_external = ftime%clock_order_external
    ctime%clock_analyse_external = ftime%clock_analyse_external
    ctime%clock_factorize_external = ftime%clock_factorize_external
    ctime%clock_solve_external = ftime%clock_solve_external
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( sls_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_sls_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%more_info = cinform%more_info
    finform%entries = cinform%entries
    finform%out_of_range = cinform%out_of_range
    finform%duplicates = cinform%duplicates
    finform%upper = cinform%upper
    finform%missing_diagonals = cinform%missing_diagonals
    finform%max_depth_assembly_tree = cinform%max_depth_assembly_tree
    finform%nodes_assembly_tree = cinform%nodes_assembly_tree
    finform%real_size_desirable = cinform%real_size_desirable
    finform%integer_size_desirable = cinform%integer_size_desirable
    finform%real_size_necessary = cinform%real_size_necessary
    finform%integer_size_necessary = cinform%integer_size_necessary
    finform%real_size_factors = cinform%real_size_factors
    finform%integer_size_factors = cinform%integer_size_factors
    finform%entries_in_factors = cinform%entries_in_factors
    finform%max_task_pool_size = cinform%max_task_pool_size
    finform%max_front_size = cinform%max_front_size
    finform%compresses_real = cinform%compresses_real
    finform%compresses_integer = cinform%compresses_integer
    finform%two_by_two_pivots = cinform%two_by_two_pivots
    finform%semi_bandwidth = cinform%semi_bandwidth
    finform%delayed_pivots = cinform%delayed_pivots
    finform%pivot_sign_changes = cinform%pivot_sign_changes
    finform%static_pivots = cinform%static_pivots
    finform%first_modified_pivot = cinform%first_modified_pivot
    finform%rank = cinform%rank
    finform%negative_eigenvalues = cinform%negative_eigenvalues
    finform%num_zero = cinform%num_zero
    finform%iterative_refinements = cinform%iterative_refinements
    finform%flops_assembly = cinform%flops_assembly
    finform%flops_elimination = cinform%flops_elimination
    finform%flops_blas = cinform%flops_blas
    finform%mc61_info = cinform%mc61_info
    finform%mc77_info = cinform%mc77_info
    finform%mumps_error = cinform%mumps_error
    finform%mumps_info = cinform%mumps_info
    finform%pardiso_error = cinform%pardiso_error
    finform%pardiso_IPARM = cinform%pardiso_IPARM
    finform%mkl_pardiso_error = cinform%mkl_pardiso_error
    finform%mkl_pardiso_IPARM = cinform%mkl_pardiso_IPARM
    finform%pastix_info = cinform%pastix_info
    finform%wsmp_error = cinform%wsmp_error
    finform%wsmp_iparm = cinform%wsmp_iparm
    finform%mpi_ierr = cinform%mpi_ierr
    finform%lapack_error = cinform%lapack_error

    ! Reals
    finform%largest_modified_pivot = cinform%largest_modified_pivot
    finform%minimum_scaling_factor = cinform%minimum_scaling_factor
    finform%maximum_scaling_factor = cinform%maximum_scaling_factor
    finform%condition_number_1 = cinform%condition_number_1
    finform%condition_number_2 = cinform%condition_number_2
    finform%backward_error_1 = cinform%backward_error_1
    finform%backward_error_2 = cinform%backward_error_2
    finform%forward_error = cinform%forward_error
    finform%mc61_rinfo = cinform%mc61_rinfo
    finform%mc77_rinfo = cinform%mc77_rinfo
    finform%mumps_rinfo = cinform%mumps_rinfo
    finform%pardiso_DPARM = cinform%pardiso_DPARM
    finform%wsmp_dparm = cinform%wsmp_dparm

    ! Logicals
    finform%alternative = cinform%alternative

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
    CALL copy_sils_ainfo_in( cinform%sils_ainfo, finform%sils_ainfo )
    CALL copy_sils_finfo_in( cinform%sils_finfo, finform%sils_finfo )
    CALL copy_sils_sinfo_in( cinform%sils_sinfo, finform%sils_sinfo )
!   CALL copy_ssids_inform_in( cinform%ssids_inform, finform%ssids_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    DO i = 1, LEN( finform%solver )
      IF ( cinform%solver( i ) == C_NULL_CHAR ) EXIT
      finform%solver( i : i ) = cinform%solver( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_sls_inform_type ), INTENT( IN ) :: finform
    TYPE ( sls_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%more_info = finform%more_info
    cinform%entries = finform%entries
    cinform%out_of_range = finform%out_of_range
    cinform%duplicates = finform%duplicates
    cinform%upper = finform%upper
    cinform%missing_diagonals = finform%missing_diagonals
    cinform%max_depth_assembly_tree = finform%max_depth_assembly_tree
    cinform%nodes_assembly_tree = finform%nodes_assembly_tree
    cinform%real_size_desirable = finform%real_size_desirable
    cinform%integer_size_desirable = finform%integer_size_desirable
    cinform%real_size_necessary = finform%real_size_necessary
    cinform%integer_size_necessary = finform%integer_size_necessary
    cinform%real_size_factors = finform%real_size_factors
    cinform%integer_size_factors = finform%integer_size_factors
    cinform%entries_in_factors = finform%entries_in_factors
    cinform%max_task_pool_size = finform%max_task_pool_size
    cinform%max_front_size = finform%max_front_size
    cinform%compresses_real = finform%compresses_real
    cinform%compresses_integer = finform%compresses_integer
    cinform%two_by_two_pivots = finform%two_by_two_pivots
    cinform%semi_bandwidth = finform%semi_bandwidth
    cinform%delayed_pivots = finform%delayed_pivots
    cinform%pivot_sign_changes = finform%pivot_sign_changes
    cinform%static_pivots = finform%static_pivots
    cinform%first_modified_pivot = finform%first_modified_pivot
    cinform%rank = finform%rank
    cinform%negative_eigenvalues = finform%negative_eigenvalues
    cinform%num_zero = finform%num_zero
    cinform%iterative_refinements = finform%iterative_refinements
    cinform%flops_assembly = finform%flops_assembly
    cinform%flops_elimination = finform%flops_elimination
    cinform%flops_blas = finform%flops_blas
    cinform%mc61_info = finform%mc61_info
    cinform%mc77_info = finform%mc77_info
    cinform%mumps_error = finform%mumps_error
    cinform%mumps_info = finform%mumps_info
    cinform%pardiso_error = finform%pardiso_error
    cinform%pardiso_IPARM = finform%pardiso_IPARM
    cinform%mkl_pardiso_error = finform%mkl_pardiso_error
    cinform%mkl_pardiso_IPARM = finform%mkl_pardiso_IPARM
    cinform%pastix_info = finform%pastix_info
    cinform%wsmp_error = finform%wsmp_error
    cinform%wsmp_iparm = finform%wsmp_iparm
    cinform%mpi_ierr = finform%mpi_ierr
    cinform%lapack_error = finform%lapack_error

    ! Reals
    cinform%largest_modified_pivot = finform%largest_modified_pivot
    cinform%minimum_scaling_factor = finform%minimum_scaling_factor
    cinform%maximum_scaling_factor = finform%maximum_scaling_factor
    cinform%condition_number_1 = finform%condition_number_1
    cinform%condition_number_2 = finform%condition_number_2
    cinform%backward_error_1 = finform%backward_error_1
    cinform%backward_error_2 = finform%backward_error_2
    cinform%forward_error = finform%forward_error
    cinform%mc61_rinfo = finform%mc61_rinfo
    cinform%mc77_rinfo = finform%mc77_rinfo
    cinform%mumps_rinfo = finform%mumps_rinfo
    cinform%pardiso_DPARM = finform%pardiso_DPARM
    cinform%wsmp_dparm = finform%wsmp_dparm

    ! Logicals
    cinform%alternative = finform%alternative

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
    CALL copy_sils_ainfo_out( finform%sils_ainfo, cinform%sils_ainfo )
    CALL copy_sils_finfo_out( finform%sils_finfo, cinform%sils_finfo )
    CALL copy_sils_sinfo_out( finform%sils_sinfo, cinform%sils_sinfo )
    CALL copy_ma57_ainfo_out( finform%ma57_ainfo, cinform%ma57_ainfo )
    CALL copy_ma57_finfo_out( finform%ma57_finfo, cinform%ma57_finfo )
    CALL copy_ma57_sinfo_out( finform%ma57_sinfo, cinform%ma57_sinfo )
    CALL copy_ma77_info_out( finform%ma77_info, cinform%ma77_info )
    CALL copy_ma86_info_out( finform%ma86_info, cinform%ma86_info )
    CALL copy_ma87_info_out( finform%ma87_info, cinform%ma87_info )
    CALL copy_ma97_info_out( finform%ma97_info, cinform%ma97_info )
    CALL copy_ssids_inform_out( finform%ssids_inform, cinform%ssids_inform )
    CALL copy_mc64_info_out( finform%mc64_info, cinform%mc64_info )
    CALL copy_mc68_info_out( finform%mc68_info, cinform%mc68_info )
    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    l = LEN( finform%solver )
    DO i = 1, l
      cinform%solver( i ) = finform%solver( i : i )
    END DO
    cinform%solver( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_SLS_precision_ciface

!  -------------------------------------
!  C interface to fortran sls_initialize
!  -------------------------------------

  SUBROUTINE sls_initialize( csolver, cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_SLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: csolver
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( sls_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_sls_full_data_type ), POINTER :: fdata
  TYPE ( f_sls_control_type ) :: fcontrol
  TYPE ( f_sls_inform_type ) :: finform
  LOGICAL :: f_indexing
  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( csolver ) ) :: fsolver

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  convert C string to Fortran string

  fsolver = cstr_to_fchar( csolver )

!  initialize required fortran types

  CALL f_sls_initialize( fsolver, fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )

  RETURN

  END SUBROUTINE sls_initialize

!  ----------------------------------------
!  C interface to fortran sls_read_specfile
!  ----------------------------------------

  SUBROUTINE sls_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_SLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( sls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_sls_control_type ) :: fcontrol
  CHARACTER ( KIND = C_CHAR, LEN = strlen( cspecfile ) ) :: fspecfile
  LOGICAL :: f_indexing

!  device unit number for specfile

  INTEGER ( KIND = ipc_ ), PARAMETER :: device = 10

!  convert C string to Fortran string

  fspecfile = cstr_to_fchar( cspecfile )

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  open specfile for reading

  OPEN( UNIT = device, FILE = fspecfile )

!  read control parameters from the specfile

  CALL f_sls_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE sls_read_specfile

!  -----------------------------------------
!  C interface to fortran sls_analyse_matrix
!  -----------------------------------------

  SUBROUTINE sls_analyse_matrix( ccontrol, cdata, status, n, ctype, ne,        &
                                 row, col, ptr  ) BIND( C )
  USE GALAHAD_SLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( sls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, ne
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ne ), OPTIONAL :: row
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ne ), OPTIONAL :: col
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: ptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: ctype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( ctype ) ) :: ftype
  TYPE ( f_sls_control_type ) :: fcontrol
  TYPE ( f_sls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  ftype = cstr_to_fchar( ctype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  analyse_matrix the problem data into the required SLS structure

  CALL f_sls_analyse_matrix( fcontrol, fdata, status, n, ftype, ne,            &
                             row, col, ptr )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE sls_analyse_matrix

!  ----------------------------------------
!  C interface to fortran sls_reset_control
!  -----------------------------------------

  SUBROUTINE sls_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_SLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( sls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_sls_control_type ) :: fcontrol
  TYPE ( f_sls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_SLS_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE sls_reset_control

!  --------------------------------------------
!  C interface to fortran sls_factorize_matrix
!  --------------------------------------------

  SUBROUTINE sls_factorize_matrix( cdata, status, ne, val ) BIND( C )
  USE GALAHAD_SLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: ne
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( ne ) :: val
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_sls_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!   form and factorize the block matrix

  CALL f_sls_factorize_matrix( fdata, status, val )
  RETURN

  END SUBROUTINE sls_factorize_matrix

!  ----------------------------------------
!  C interface to fortran sls_solve_system
!  ----------------------------------------

  SUBROUTINE sls_solve_system( cdata, status, n, sol ) BIND( C )
  USE GALAHAD_SLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: sol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_sls_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!   form and factorize the block matrix

  CALL f_sls_solve_system( fdata, status, sol )
  RETURN

  END SUBROUTINE sls_solve_system

!  ----------------------------------------
!  C interface to fortran sls_partial_solve_system
!  ----------------------------------------

  SUBROUTINE sls_partial_solve_system( cpart, cdata, status, n, sol ) BIND( C )
  USE GALAHAD_SLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: sol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cpart
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_sls_full_data_type ), POINTER :: fdata
  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( cpart ) ) :: fpart

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  fpart = cstr_to_fchar( cpart )

!   form and factorize the block matrix

  CALL f_sls_partial_solve( fpart, fdata, status, sol )
  RETURN

  END SUBROUTINE sls_partial_solve_system

!  --------------------------------------
!  C interface to fortran sls_information
!  --------------------------------------

  SUBROUTINE sls_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_SLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( sls_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_sls_full_data_type ), pointer :: fdata
  TYPE ( f_sls_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain SLS solution information

  CALL f_sls_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE sls_information

!  ------------------------------------
!  C interface to fortran sls_terminate
!  ------------------------------------

  SUBROUTINE sls_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_SLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( sls_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( sls_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_sls_full_data_type ), pointer :: fdata
  TYPE ( f_sls_control_type ) :: fcontrol
  TYPE ( f_sls_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_sls_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE sls_terminate
