! THIS VERSION: GALAHAD 3.3 - 24/12/2021 AT 09:22 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _  D Q P    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.3. December 24th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_DQP_double_ciface
    USE iso_c_binding
    USE GALAHAD_common_ciface
    USE GALAHAD_DQP_double, ONLY:                                              &
        f_dqp_control_type   => DQP_control_type,                              &
        f_dqp_time_type      => DQP_time_type,                                 &
        f_dqp_inform_type    => DQP_inform_type,                               &
        f_dqp_full_data_type => DQP_full_data_type,                            &
        f_dqp_initialize     => DQP_initialize,                                &
        f_dqp_read_specfile  => DQP_read_specfile,                             &
        f_dqp_import         => DQP_import,                                    &
        f_dqp_reset_control  => DQP_reset_control,                             &
        f_dqp_solve_qp       => DQP_solve_qp,                                  &
        f_dqp_solve_sldqp    => DQP_solve_sldqp,                               &
        f_dqp_information    => DQP_information,                               &
        f_dqp_terminate      => DQP_terminate

!    USE GALAHAD_FDC_double_ciface, ONLY: &
!        fdc_inform_type, &
!        fdc_control_type, &
!        copy_fdc_inform_in => copy_inform_in, &
!        copy_fdc_inform_out => copy_inform_out, &
!        copy_fdc_control_in => copy_control_in, &
!        copy_fdc_control_out => copy_control_out

    USE GALAHAD_SLS_double_ciface, ONLY:                                       &
        sls_inform_type,                                                       &
        sls_control_type,                                                      &
        copy_sls_inform_in   => copy_inform_in,                                &
        copy_sls_inform_out  => copy_inform_out,                               &
        copy_sls_control_in  => copy_control_in,                               &
        copy_sls_control_out => copy_control_out

    USE GALAHAD_SBLS_double_ciface, ONLY:                                      &
        sbls_inform_type,                                                      &
        sbls_control_type,                                                     &
        copy_sbls_inform_in   => copy_inform_in,                               &
        copy_sbls_inform_out  => copy_inform_out,                              &
        copy_sbls_control_in  => copy_control_in,                              &
        copy_sbls_control_out => copy_control_out

    USE GALAHAD_GLTR_double_ciface, ONLY:                                      &
         gltr_inform_type,                                                     &
         gltr_control_type,                                                    &
         copy_gltr_inform_in   => copy_inform_in,                              &
         copy_gltr_inform_out  => copy_inform_out,                             &
         copy_gltr_control_in  => copy_control_in,                             &
         copy_gltr_control_out => copy_control_out

!    USE GALAHAD_SCU_double_ciface, ONLY: &
!        scu_inform_type, &
!        scu_control_type, &
!        copy_scu_inform_in => copy_inform_in, &
!        copy_scu_inform_out => copy_inform_out, &
!        copy_scu_control_in => copy_control_in, &
!        copy_scu_control_out => copy_control_out

!    USE GALAHAD_RPD_double_ciface, ONLY: &
!        rpd_inform_type, &
!        rpd_control_type, &
!        copy_rpd_inform_in => copy_inform_in, &
!        copy_rpd_inform_out => copy_inform_out, &
!        copy_rpd_control_in => copy_control_in, &
!        copy_rpd_control_out => copy_control_out

    IMPLICIT NONE

!--------------------
!   P r e c i s i o n
!--------------------

    INTEGER, PARAMETER :: wp = C_DOUBLE ! double precision
    INTEGER, PARAMETER :: sp = C_FLOAT  ! single precision

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: dqp_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = C_INT ) :: error
      INTEGER ( KIND = C_INT ) :: out
      INTEGER ( KIND = C_INT ) :: print_level
      INTEGER ( KIND = C_INT ) :: start_print
      INTEGER ( KIND = C_INT ) :: stop_print
      INTEGER ( KIND = C_INT ) :: print_gap
      INTEGER ( KIND = C_INT ) :: dual_starting_point
      INTEGER ( KIND = C_INT ) :: maxit
      INTEGER ( KIND = C_INT ) :: max_sc
      INTEGER ( KIND = C_INT ) :: cauchy_only
      INTEGER ( KIND = C_INT ) :: arc_search_maxit
      INTEGER ( KIND = C_INT ) :: cg_maxit
      INTEGER ( KIND = C_INT ) :: explore_optimal_subspace
      INTEGER ( KIND = C_INT ) :: restore_problem
      INTEGER ( KIND = C_INT ) :: sif_file_device
      INTEGER ( KIND = C_INT ) :: qplib_file_device
      REAL ( KIND = wp ) :: rho
      REAL ( KIND = wp ) :: infinity
      REAL ( KIND = wp ) :: stop_abs_p
      REAL ( KIND = wp ) :: stop_rel_p
      REAL ( KIND = wp ) :: stop_abs_d
      REAL ( KIND = wp ) :: stop_rel_d
      REAL ( KIND = wp ) :: stop_abs_c
      REAL ( KIND = wp ) :: stop_rel_c
      REAL ( KIND = wp ) :: stop_cg_relative
      REAL ( KIND = wp ) :: stop_cg_absolute
      REAL ( KIND = wp ) :: cg_zero_curvature
      REAL ( KIND = wp ) :: max_growth
      REAL ( KIND = wp ) :: identical_bounds_tol
      REAL ( KIND = wp ) :: cpu_time_limit
      REAL ( KIND = wp ) :: clock_time_limit
      REAL ( KIND = wp ) :: initial_perturbation
      REAL ( KIND = wp ) :: perturbation_reduction
      REAL ( KIND = wp ) :: final_perturbation
      LOGICAL ( KIND = C_BOOL ) :: factor_optimal_matrix
      LOGICAL ( KIND = C_BOOL ) :: remove_dependencies
      LOGICAL ( KIND = C_BOOL ) :: treat_zero_bounds_as_general
      LOGICAL ( KIND = C_BOOL ) :: exact_arc_search
      LOGICAL ( KIND = C_BOOL ) :: subspace_direct
      LOGICAL ( KIND = C_BOOL ) :: subspace_alternate
      LOGICAL ( KIND = C_BOOL ) :: subspace_arc_search
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      LOGICAL ( KIND = C_BOOL ) :: generate_sif_file
      LOGICAL ( KIND = C_BOOL ) :: generate_qplib_file
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: symmetric_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: definite_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: unsymmetric_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: sif_file_name
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: qplib_file_name
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
!     TYPE ( fdc_control_type ) :: fdc_control
      TYPE ( sls_control_type ) :: sls_control
      TYPE ( sbls_control_type ) :: sbls_control
      TYPE ( gltr_control_type ) :: gltr_control
    END TYPE dqp_control_type

    TYPE, BIND( C ) :: dqp_time_type
      REAL ( KIND = wp ) :: total
      REAL ( KIND = wp ) :: preprocess
      REAL ( KIND = wp ) :: find_dependent
      REAL ( KIND = wp ) :: analyse
      REAL ( KIND = wp ) :: factorize
      REAL ( KIND = wp ) :: solve
      REAL ( KIND = wp ) :: search
      REAL ( KIND = wp ) :: clock_total
      REAL ( KIND = wp ) :: clock_preprocess
      REAL ( KIND = wp ) :: clock_find_dependent
      REAL ( KIND = wp ) :: clock_analyse
      REAL ( KIND = wp ) :: clock_factorize
      REAL ( KIND = wp ) :: clock_solve
      REAL ( KIND = wp ) :: clock_search
    END TYPE dqp_time_type

    TYPE, BIND( C ) :: dqp_inform_type
      INTEGER ( KIND = C_INT ) :: status
      INTEGER ( KIND = C_INT ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = C_INT ) :: iter
      INTEGER ( KIND = C_INT ) :: cg_iter
      INTEGER ( KIND = C_INT ) :: factorization_status
      INTEGER ( KIND = C_INT ) :: factorization_integer
      INTEGER ( KIND = C_INT ) :: factorization_real
      INTEGER ( KIND = C_INT ) :: nfacts
      INTEGER ( KIND = C_INT ) :: threads
      REAL ( KIND = wp ) :: obj
      REAL ( KIND = wp ) :: primal_infeasibility
      REAL ( KIND = wp ) :: dual_infeasibility
      REAL ( KIND = wp ) :: complementary_slackness
      REAL ( KIND = wp ) :: non_negligible_pivot
      LOGICAL ( KIND = C_BOOL ) :: feasible
      INTEGER ( KIND = C_INT ), DIMENSION( 16 ) :: checkpointsIter
      REAL ( KIND = wp ), DIMENSION( 16 ) :: checkpointsTime
      TYPE ( dqp_time_type ) :: time
!     TYPE ( fdc_inform_type ) :: fdc_inform
      TYPE ( sls_inform_type ) :: sls_inform
      TYPE ( sbls_inform_type ) :: sbls_inform
      TYPE ( gltr_inform_type ) :: gltr_inform
      INTEGER ( KIND = C_INT ) :: scu_status
!     TYPE ( scu_inform_type ) :: scu_inform
!     TYPE ( rpd_inform_type ) :: rpd_inform
    END TYPE dqp_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing ) 
    TYPE ( dqp_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_dqp_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, optional, INTENT( OUT ) :: f_indexing
    INTEGER :: i
    
    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%start_print = ccontrol%start_print
    fcontrol%stop_print = ccontrol%stop_print
    fcontrol%print_gap = ccontrol%print_gap
    fcontrol%dual_starting_point = ccontrol%dual_starting_point
    fcontrol%maxit = ccontrol%maxit
    fcontrol%max_sc = ccontrol%max_sc
    fcontrol%cauchy_only = ccontrol%cauchy_only
    fcontrol%arc_search_maxit = ccontrol%arc_search_maxit
    fcontrol%cg_maxit = ccontrol%cg_maxit
    fcontrol%explore_optimal_subspace = ccontrol%explore_optimal_subspace
    fcontrol%restore_problem = ccontrol%restore_problem
    fcontrol%sif_file_device = ccontrol%sif_file_device
    fcontrol%qplib_file_device = ccontrol%qplib_file_device

    ! Reals
    fcontrol%rho = ccontrol%rho
    fcontrol%infinity = ccontrol%infinity
    fcontrol%stop_abs_p = ccontrol%stop_abs_p
    fcontrol%stop_rel_p = ccontrol%stop_rel_p
    fcontrol%stop_abs_d = ccontrol%stop_abs_d
    fcontrol%stop_rel_d = ccontrol%stop_rel_d
    fcontrol%stop_abs_c = ccontrol%stop_abs_c
    fcontrol%stop_rel_c = ccontrol%stop_rel_c
    fcontrol%stop_cg_relative = ccontrol%stop_cg_relative
    fcontrol%stop_cg_absolute = ccontrol%stop_cg_absolute
    fcontrol%cg_zero_curvature = ccontrol%cg_zero_curvature
    fcontrol%max_growth = ccontrol%max_growth
    fcontrol%identical_bounds_tol = ccontrol%identical_bounds_tol
    fcontrol%cpu_time_limit = ccontrol%cpu_time_limit
    fcontrol%clock_time_limit = ccontrol%clock_time_limit
    fcontrol%initial_perturbation = ccontrol%initial_perturbation
    fcontrol%perturbation_reduction = ccontrol%perturbation_reduction
    fcontrol%final_perturbation = ccontrol%final_perturbation

    ! Logicals
    fcontrol%factor_optimal_matrix = ccontrol%factor_optimal_matrix
    fcontrol%remove_dependencies = ccontrol%remove_dependencies
    fcontrol%treat_zero_bounds_as_general = ccontrol%treat_zero_bounds_as_general
    fcontrol%exact_arc_search = ccontrol%exact_arc_search
    fcontrol%subspace_direct = ccontrol%subspace_direct
    fcontrol%subspace_alternate = ccontrol%subspace_alternate
    fcontrol%subspace_arc_search = ccontrol%subspace_arc_search
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal
    fcontrol%generate_sif_file = ccontrol%generate_sif_file
    fcontrol%generate_qplib_file = ccontrol%generate_qplib_file

    ! Derived types
!   CALL copy_fdc_control_in( ccontrol%fdc_control, fcontrol%fdc_control )
    CALL copy_sls_control_in( ccontrol%sls_control, fcontrol%sls_control )
    CALL copy_sbls_control_in( ccontrol%sbls_control, fcontrol%sbls_control )
    CALL copy_gltr_control_in( ccontrol%gltr_control, fcontrol%gltr_control )

    ! Strings
    DO i = 1, LEN( fcontrol%symmetric_linear_solver )
      IF ( ccontrol%symmetric_linear_solver( i ) == C_NULL_CHAR ) EXIT
      fcontrol%symmetric_linear_solver( i : i ) = ccontrol%symmetric_linear_solver( i )
    END DO
    DO i = 1, LEN( fcontrol%definite_linear_solver )
      IF ( ccontrol%definite_linear_solver( i ) == C_NULL_CHAR ) EXIT
      fcontrol%definite_linear_solver( i : i )                                 &
        = ccontrol%definite_linear_solver( i )
    END DO
    DO i = 1, LEN( fcontrol%unsymmetric_linear_solver )
      IF ( ccontrol%unsymmetric_linear_solver( i ) == C_NULL_CHAR ) EXIT
      fcontrol%unsymmetric_linear_solver( i : i )                              &
        = ccontrol%unsymmetric_linear_solver( i )
    END DO
    DO i = 1, LEN( fcontrol%sif_file_name )
      IF ( ccontrol%sif_file_name( i ) == C_NULL_CHAR ) EXIT
      fcontrol%sif_file_name( i : i ) = ccontrol%sif_file_name( i )
    END DO
    DO i = 1, LEN( fcontrol%qplib_file_name )
      IF ( ccontrol%qplib_file_name( i ) == C_NULL_CHAR ) EXIT
      fcontrol%qplib_file_name( i : i ) = ccontrol%qplib_file_name( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing ) 
    TYPE ( f_dqp_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( dqp_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER :: i, l
    
    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%start_print = fcontrol%start_print
    ccontrol%stop_print = fcontrol%stop_print
    ccontrol%print_gap = fcontrol%print_gap
    ccontrol%dual_starting_point = fcontrol%dual_starting_point
    ccontrol%maxit = fcontrol%maxit
    ccontrol%max_sc = fcontrol%max_sc
    ccontrol%cauchy_only = fcontrol%cauchy_only
    ccontrol%arc_search_maxit = fcontrol%arc_search_maxit
    ccontrol%cg_maxit = fcontrol%cg_maxit
    ccontrol%explore_optimal_subspace = fcontrol%explore_optimal_subspace
    ccontrol%restore_problem = fcontrol%restore_problem
    ccontrol%sif_file_device = fcontrol%sif_file_device
    ccontrol%qplib_file_device = fcontrol%qplib_file_device

    ! Reals
    ccontrol%rho = fcontrol%rho
    ccontrol%infinity = fcontrol%infinity
    ccontrol%stop_abs_p = fcontrol%stop_abs_p
    ccontrol%stop_rel_p = fcontrol%stop_rel_p
    ccontrol%stop_abs_d = fcontrol%stop_abs_d
    ccontrol%stop_rel_d = fcontrol%stop_rel_d
    ccontrol%stop_abs_c = fcontrol%stop_abs_c
    ccontrol%stop_rel_c = fcontrol%stop_rel_c
    ccontrol%stop_cg_relative = fcontrol%stop_cg_relative
    ccontrol%stop_cg_absolute = fcontrol%stop_cg_absolute
    ccontrol%cg_zero_curvature = fcontrol%cg_zero_curvature
    ccontrol%max_growth = fcontrol%max_growth
    ccontrol%identical_bounds_tol = fcontrol%identical_bounds_tol
    ccontrol%cpu_time_limit = fcontrol%cpu_time_limit
    ccontrol%clock_time_limit = fcontrol%clock_time_limit
    ccontrol%initial_perturbation = fcontrol%initial_perturbation
    ccontrol%perturbation_reduction = fcontrol%perturbation_reduction
    ccontrol%final_perturbation = fcontrol%final_perturbation

    ! Logicals
    ccontrol%factor_optimal_matrix = fcontrol%factor_optimal_matrix
    ccontrol%remove_dependencies = fcontrol%remove_dependencies
    ccontrol%treat_zero_bounds_as_general                                      &
      = fcontrol%treat_zero_bounds_as_general
    ccontrol%exact_arc_search = fcontrol%exact_arc_search
    ccontrol%subspace_direct = fcontrol%subspace_direct
    ccontrol%subspace_alternate = fcontrol%subspace_alternate
    ccontrol%subspace_arc_search = fcontrol%subspace_arc_search
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal
    ccontrol%generate_sif_file = fcontrol%generate_sif_file
    ccontrol%generate_qplib_file = fcontrol%generate_qplib_file

    ! Derived types
!   CALL copy_fdc_control_out( fcontrol%fdc_control, ccontrol%fdc_control )
    CALL copy_sls_control_out( fcontrol%sls_control, ccontrol%sls_control )
    CALL copy_sbls_control_out( fcontrol%sbls_control, ccontrol%sbls_control )
    CALL copy_gltr_control_out( fcontrol%gltr_control, ccontrol%gltr_control )

    ! Strings
    l = LEN( fcontrol%symmetric_linear_solver )
    DO i = 1, l
      ccontrol%symmetric_linear_solver( i )                                    &
        = fcontrol%symmetric_linear_solver( i : i )
    END DO
    ccontrol%symmetric_linear_solver( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%definite_linear_solver )
    DO i = 1, l
      ccontrol%definite_linear_solver( i )                                     &
        = fcontrol%definite_linear_solver( i : i )
    END DO
    ccontrol%definite_linear_solver( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%unsymmetric_linear_solver )
    DO i = 1, l
      ccontrol%unsymmetric_linear_solver( i )                                  &
        = fcontrol%unsymmetric_linear_solver( i : i )
    END DO
    ccontrol%unsymmetric_linear_solver( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%sif_file_name )
    DO i = 1, l
      ccontrol%sif_file_name( i ) = fcontrol%sif_file_name( i : i )
    END DO
    ccontrol%sif_file_name( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%qplib_file_name )
    DO i = 1, l
      ccontrol%qplib_file_name( i ) = fcontrol%qplib_file_name( i : i )
    END DO
    ccontrol%qplib_file_name( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime ) 
    TYPE ( dqp_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_dqp_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%preprocess = ctime%preprocess
    ftime%find_dependent = ctime%find_dependent
    ftime%analyse = ctime%analyse
    ftime%factorize = ctime%factorize
    ftime%solve = ctime%solve
    ftime%search = ctime%search
    ftime%clock_total = ctime%clock_total
    ftime%clock_preprocess = ctime%clock_preprocess
    ftime%clock_find_dependent = ctime%clock_find_dependent
    ftime%clock_analyse = ctime%clock_analyse
    ftime%clock_factorize = ctime%clock_factorize
    ftime%clock_solve = ctime%clock_solve
    ftime%clock_search = ctime%clock_search
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime ) 
    TYPE ( f_dqp_time_type ), INTENT( IN ) :: ftime
    TYPE ( dqp_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%preprocess = ftime%preprocess
    ctime%find_dependent = ftime%find_dependent
    ctime%analyse = ftime%analyse
    ctime%factorize = ftime%factorize
    ctime%solve = ftime%solve
    ctime%search = ftime%search
    ctime%clock_total = ftime%clock_total
    ctime%clock_preprocess = ftime%clock_preprocess
    ctime%clock_find_dependent = ftime%clock_find_dependent
    ctime%clock_analyse = ftime%clock_analyse
    ctime%clock_factorize = ftime%clock_factorize
    ctime%clock_solve = ftime%clock_solve
    ctime%clock_search = ftime%clock_search
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform ) 
    TYPE ( dqp_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_dqp_inform_type ), INTENT( OUT ) :: finform
    INTEGER :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%iter = cinform%iter
    finform%cg_iter = cinform%cg_iter
    finform%factorization_status = cinform%factorization_status
    finform%factorization_integer = cinform%factorization_integer
    finform%factorization_real = cinform%factorization_real
    finform%nfacts = cinform%nfacts
    finform%threads = cinform%threads
    finform%checkpointsIter = cinform%checkpointsIter
    finform%scu_status = cinform%scu_status

    ! Reals
    finform%obj = cinform%obj
    finform%primal_infeasibility = cinform%primal_infeasibility
    finform%dual_infeasibility = cinform%dual_infeasibility
    finform%complementary_slackness = cinform%complementary_slackness
    finform%non_negligible_pivot = cinform%non_negligible_pivot
    finform%checkpointsTime = cinform%checkpointsTime

    ! Logicals
    finform%feasible = cinform%feasible

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
!   CALL copy_fdc_inform_in( cinform%fdc_inform, finform%fdc_inform )
    CALL copy_sls_inform_in( cinform%sls_inform, finform%sls_inform )
    CALL copy_sbls_inform_in( cinform%sbls_inform, finform%sbls_inform )
    CALL copy_gltr_inform_in( cinform%gltr_inform, finform%gltr_inform )
!   CALL copy_scu_inform_in( cinform%scu_inform, finform%scu_inform )
!   CALL copy_rpd_inform_in( cinform%rpd_inform, finform%rpd_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform ) 
    TYPE ( f_dqp_inform_type ), INTENT( IN ) :: finform
    TYPE ( dqp_inform_type ), INTENT( OUT ) :: cinform
    INTEGER :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%iter = finform%iter
    cinform%cg_iter = finform%cg_iter
    cinform%factorization_status = finform%factorization_status
    cinform%factorization_integer = finform%factorization_integer
    cinform%factorization_real = finform%factorization_real
    cinform%nfacts = finform%nfacts
    cinform%threads = finform%threads
    cinform%checkpointsIter = finform%checkpointsIter
    cinform%scu_status = finform%scu_status

    ! Reals
    cinform%obj = finform%obj
    cinform%primal_infeasibility = finform%primal_infeasibility
    cinform%dual_infeasibility = finform%dual_infeasibility
    cinform%complementary_slackness = finform%complementary_slackness
    cinform%non_negligible_pivot = finform%non_negligible_pivot
    cinform%checkpointsTime = finform%checkpointsTime

    ! Logicals
    cinform%feasible = finform%feasible

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
!   CALL copy_fdc_inform_out( finform%fdc_inform, cinform%fdc_inform )
    CALL copy_sls_inform_out( finform%sls_inform, cinform%sls_inform )
    CALL copy_sbls_inform_out( finform%sbls_inform, cinform%sbls_inform )
    CALL copy_gltr_inform_out( finform%gltr_inform, cinform%gltr_inform )
!   CALL copy_scu_inform_out( finform%scu_inform, cinform%scu_inform )
!   CALL copy_rpd_inform_out( finform%rpd_inform, cinform%rpd_inform )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_DQP_double_ciface

!  -------------------------------------
!  C interface to fortran dqp_initialize
!  -------------------------------------

  SUBROUTINE dqp_initialize( cdata, ccontrol, status ) BIND( C ) 
  USE GALAHAD_DQP_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( dqp_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_dqp_full_data_type ), POINTER :: fdata
  TYPE ( f_dqp_control_type ) :: fcontrol
  TYPE ( f_dqp_inform_type ) :: finform
  LOGICAL :: f_indexing 

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_dqp_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out 

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE dqp_initialize

!  ----------------------------------------
!  C interface to fortran dqp_read_specfile
!  ----------------------------------------

  SUBROUTINE dqp_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_DQP_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( dqp_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_dqp_control_type ) :: fcontrol
  CHARACTER ( KIND = C_CHAR, LEN = strlen( cspecfile ) ) :: fspecfile
  LOGICAL :: f_indexing

!  device unit number for specfile

  INTEGER ( KIND = C_INT ), PARAMETER :: device = 10

!  convert C string to Fortran string

  fspecfile = cstr_to_fchar( cspecfile )

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )
  
!  open specfile for reading

  OPEN( UNIT = device, FILE = fspecfile )
  
!  read control parameters from the specfile

  CALL f_dqp_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE dqp_read_specfile

!  ---------------------------------
!  C interface to fortran dqp_inport
!  ---------------------------------

  SUBROUTINE dqp_import( ccontrol, cdata, status, n, m,                        &
                         chtype, hne, hrow, hcol, hptr,                        &
                         catype, ane, arow, acol, aptr ) BIND( C )
  USE GALAHAD_DQP_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  TYPE ( dqp_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n, m, hne, ane
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hrow
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hcol
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: hptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: chtype
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: arow
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: acol
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( m + 1 ), OPTIONAL :: aptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: catype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( chtype ) ) :: fhtype
  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( catype ) ) :: fatype
  TYPE ( f_dqp_control_type ) :: fcontrol
  TYPE ( f_dqp_full_data_type ), POINTER :: fdata
  INTEGER, DIMENSION( : ), ALLOCATABLE :: hrow_find, hcol_find, hptr_find
  INTEGER, DIMENSION( : ), ALLOCATABLE :: arow_find, acol_find, aptr_find
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  fhtype = cstr_to_fchar( chtype )
  fatype = cstr_to_fchar( catype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  handle C sparse matrix indexing

  IF ( .NOT. f_indexing ) THEN
    IF ( PRESENT( hrow ) ) THEN
      ALLOCATE( hrow_find( hne ) )
      hrow_find = hrow + 1
    END IF
    IF ( PRESENT( hcol ) ) THEN
      ALLOCATE( hcol_find( hne ) )
      hcol_find = hcol + 1
    END IF
    IF ( PRESENT( hptr ) ) THEN
      ALLOCATE( hptr_find( n + 1 ) )
      hptr_find = hptr + 1
    END IF

    IF ( PRESENT( arow ) ) THEN
      ALLOCATE( arow_find( ane ) )
      arow_find = arow + 1
    END IF
    IF ( PRESENT( acol ) ) THEN
      ALLOCATE( acol_find( ane ) )
      acol_find = acol + 1
    END IF
    IF ( PRESENT( aptr ) ) THEN
      ALLOCATE( aptr_find( m + 1 ) )
      aptr_find = aptr + 1
    END IF

!  import the problem data into the required DQP structure

    CALL f_dqp_import( fcontrol, fdata, status, n, m,                          &
                       fhtype, hne, hrow_find, hcol_find, hptr_find,           &
                       fatype, ane, arow_find, acol_find, aptr_find )

    IF ( ALLOCATED( hrow_find ) ) DEALLOCATE( hrow_find )
    IF ( ALLOCATED( hcol_find ) ) DEALLOCATE( hcol_find )
    IF ( ALLOCATED( hptr_find ) ) DEALLOCATE( hptr_find )
    IF ( ALLOCATED( arow_find ) ) DEALLOCATE( arow_find )
    IF ( ALLOCATED( acol_find ) ) DEALLOCATE( acol_find )
    IF ( ALLOCATED( aptr_find ) ) DEALLOCATE( aptr_find )
  ELSE
    CALL f_dqp_import( fcontrol, fdata, status, n, m,                          &
                       fhtype, hne, hrow, hcol, hptr,                          &
                       fatype, ane, arow, acol, aptr )
  END IF

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE dqp_import

!  ----------------------------------------
!  C interface to fortran dqp_reset_control
!  ----------------------------------------

  SUBROUTINE dqp_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_DQP_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  TYPE ( dqp_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_dqp_control_type ) :: fcontrol
  TYPE ( f_dqp_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_dqp_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE dqp_reset_control

!  ------------------------------------
!  C interface to fortran dqp_solve_dqp
!  ------------------------------------

  SUBROUTINE dqp_solve_qp( cdata, status, n, m, hne, hval, g, f, ane, aval,    &
                           cl, cu, xl, xu, x, c, y, z, xstat, cstat ) BIND( C )
  USE GALAHAD_DQP_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n, m, ane, hne
  INTEGER ( KIND = C_INT ), INTENT( INOUT ) :: status
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( hne ) :: hval
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ane ) :: aval
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: g
  REAL ( KIND = wp ), INTENT( IN ), VALUE :: f
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: cl, cu
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: xl, xu
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: x, z
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: y
  REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: c
  INTEGER ( KIND = C_INT ), INTENT( OUT ), DIMENSION( n ) :: xstat
  INTEGER ( KIND = C_INT ), INTENT( OUT ), DIMENSION( m ) :: cstat
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_dqp_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_dqp_solve_qp( fdata, status, hval, g, f, aval, cl, cu, xl, xu,        &
                       x, c, y, z, xstat, cstat )
  RETURN

  END SUBROUTINE dqp_solve_qp

!  --------------------------------------
!  C interface to fortran dqp_solve_sldqp
!  --------------------------------------

  SUBROUTINE dqp_solve_sldqp( cdata, status, n, m, w, x0, g, f, ane, aval, cl, &
                              cu, xl, xu, x, c, y, z, xstat, cstat ) BIND( C )
  USE GALAHAD_DQP_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n, m, ane
  INTEGER ( KIND = C_INT ), INTENT( INOUT ) :: status
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: w
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: x0
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ane ) :: aval
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: g
  REAL ( KIND = wp ), INTENT( IN ), VALUE :: f
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: cl, cu
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: xl, xu
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: x, z
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: y
  REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: c
  INTEGER ( KIND = C_INT ), INTENT( OUT ), DIMENSION( n ) :: xstat
  INTEGER ( KIND = C_INT ), INTENT( OUT ), DIMENSION( m ) :: cstat
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_dqp_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_dqp_solve_sldqp( fdata, status, w, x0, g, f, aval, cl, cu, xl, xu,    &
                          x, c, y, z, xstat, cstat )
  RETURN

  END SUBROUTINE dqp_solve_sldqp

!  --------------------------------------
!  C interface to fortran dqp_information
!  --------------------------------------

  SUBROUTINE dqp_information( cdata, cinform, status ) BIND( C ) 
  USE GALAHAD_DQP_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( dqp_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_dqp_full_data_type ), pointer :: fdata
  TYPE ( f_dqp_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain DQP solution information

  CALL f_dqp_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE dqp_information

!  ------------------------------------
!  C interface to fortran dqp_terminate
!  ------------------------------------

  SUBROUTINE dqp_terminate( cdata, ccontrol, cinform ) BIND( C ) 
  USE GALAHAD_DQP_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( dqp_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( dqp_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_dqp_full_data_type ), pointer :: fdata
  TYPE ( f_dqp_control_type ) :: fcontrol
  TYPE ( f_dqp_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_dqp_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR 
  RETURN

  END SUBROUTINE dqp_terminate
